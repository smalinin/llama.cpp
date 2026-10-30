#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json       = nlohmann::json;
using clock_type = std::chrono::steady_clock;

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static double elapsed(clock_type::time_point begin) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - begin).count();
}

static json stats(const std::vector<double> & values) {
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    return {
        { "samples_us", values                    },
        { "median_us",  sorted[sorted.size() / 2] },
        { "min_us",     sorted.front()            },
        { "max_us",     sorted.back()             }
    };
}

static void save(const std::string & path, const json & value) {
    std::ofstream file(path);
    file << value.dump(2) << '\n';
    require(bool(file), "cannot write probe output");
}

struct mapping {
    int                   fd     = -1;
    void *                data   = MAP_FAILED;
    size_t                size   = 0;
    ggml_backend_buffer_t buffer = nullptr;

    void open(const std::string & path) {
        fd = ::open(path.c_str(), O_RDONLY);
        require(fd >= 0, "cannot open weights");

        struct stat st {};

        require(fstat(fd, &st) == 0 && st.st_size > 0, "cannot stat weights");
        size = st.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        require(data != MAP_FAILED, "cannot map weights");
        buffer = ggml_backend_cpu_buffer_from_ptr(data, size);
        require(buffer, "cannot wrap mmap");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    ~mapping() {
        ggml_backend_buffer_free(buffer);
        if (data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

struct bench {
    std::map<std::string, std::unique_ptr<mapping>> maps;
    ggml_context_ptr                                host{ nullptr }, device{ nullptr };
    ggml_backend_ptr                                cpu{ nullptr }, gpu{ nullptr };
    ggml_backend_buffer_ptr                         weights{ nullptr };
    ggml_backend_sched_ptr                          sched{ nullptr };
    ggml_tensor *                                   h[3]{}, *g[3]{};
    int                                             dim = 0, ff = 0, experts = 0, mask = 6;
    json                                            config;

    void init(const json & cfg) {
        config = cfg;
        mask   = cfg.at("host_mask");
        require(mask > 0 && mask < 8, "invalid host mask");
        ggml_backend_load_all();
        cpu.reset(ggml_backend_cpu_init());
        gpu.reset(ggml_backend_init_by_name("CUDA0", nullptr));
        require(cpu && gpu, "CPU/CUDA0 required");
        int threads = cfg.at("threads");
        require(threads > 0 && threads <= 1024, "invalid CPU threads");
        ggml_backend_cpu_set_n_threads(cpu.get(), threads);
        host.reset(ggml_init({ 1024 * 1024, nullptr, true }));
        device.reset(ggml_init({ 1024 * 1024, nullptr, true }));
        require(host && device, "metadata allocation failed");
        require(cfg.at("tensors").size() == 3, "up/gate/down descriptors required");
        for (int i = 0; i < 3; ++i) {
            const auto & desc = cfg["tensors"][i];
            auto         ne   = desc.at("shape").get<std::vector<int64_t>>();
            require(ne.size() == 4 && ne[3] == 1 && ne[0] > 0 && ne[0] <= 65536 && ne[1] > 0 && ne[1] <= 65536 &&
                        ne[2] > 0 && ne[2] <= 4096,
                    "invalid tensor shape");
            int type = desc.at("type_id");
            require(type >= 0 && type < GGML_TYPE_COUNT && ggml_is_quantized(ggml_type(type)),
                    "quantized weights required");
            h[i]                   = ggml_new_tensor_3d(host.get(), ggml_type(type), ne[0], ne[1], ne[2]);
            g[i]                   = ggml_new_tensor_3d(device.get(), ggml_type(type), ne[0], ne[1], ne[2]);
            const std::string path = desc.at("path");
            auto &            map  = maps[path];
            if (!map) {
                map.reset(new mapping);
                map->open(path);
            }
            size_t offset = desc.at("offset"), bytes = desc.at("bytes");
            require(bytes == ggml_nbytes(h[i]) && offset <= map->size && bytes <= map->size - offset,
                    "tensor bounds/size mismatch");
            h[i]->buffer = map->buffer;
            h[i]->data   = static_cast<char *>(map->data) + offset;
            ggml_set_name(h[i], ("calibrate_host_" + std::to_string(i)).c_str());
            ggml_set_name(g[i], ("calibrate_gpu_" + std::to_string(i)).c_str());
        }
        dim     = h[0]->ne[0];
        ff      = h[0]->ne[1];
        experts = h[0]->ne[2];
        require(h[1]->ne[0] == dim && h[1]->ne[1] == ff && h[1]->ne[2] == experts && h[2]->ne[0] == ff &&
                    h[2]->ne[1] == dim && h[2]->ne[2] == experts,
                "projection geometry mismatch");
        weights.reset(ggml_backend_alloc_ctx_tensors(device.get(), gpu.get()));
        require(bool(weights), "GPU weights allocation failed");
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_t backends[] = { gpu.get(), cpu.get() };
        sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 8192, false, true));
        require(bool(sched), "scheduler allocation failed");
        std::set<int> used;
        int           repeats = cfg.at("repeats"), warmup = cfg.at("warmup");
        require(repeats >= 3 && repeats <= 100 && warmup >= 1 && warmup <= 10, "invalid repetitions");
        require(!cfg.at("cases").empty(), "calibration cases required");
        for (const auto & c : cfg.at("cases")) {
            int n = c.at("tokens"), k = c.at("picks");
            require(n > 0 && n <= 512 && k > 0 && k <= experts, "invalid calibration case");
            for (int r = 0; r < repeats + warmup; ++r) {
                for (int t = 0; t < n; ++t) {
                    for (int j = 0; j < k; ++j) {
                        used.insert((r * 47 + t * k + j) % experts);
                    }
                }
            }
        }
        size_t bytes = 0;
        for (int i = 0; i < 3; ++i) {
            for (int e : used) {
                ggml_backend_tensor_set(g[i], static_cast<const char *>(h[i]->data) + e * h[i]->nb[2], e * g[i]->nb[2],
                                        h[i]->nb[2]);
                bytes += h[i]->nb[2];
            }
        }
        config["initial_upload_bytes"] = bytes;
    }

    json measure(int n, int k) {
        json result = {
            { "tokens", n              },
            { "picks",  k              },
            { "modes",  json::object() }
        };
        std::vector<std::vector<float>> reference;
        const int                       count = int(config["repeats"]) + int(config["warmup"]);
        for (const std::string mode : { "gpu_resident", "cpu_lane", "serial_h2d_gpu" }) {
            ggml_backend_sched_reset(sched.get());
            llm_graph_result res(8192);
            res.reset();
            llama_adapter_loras loras;
            llm_graph_params    p{};
            p.arch                = LLM_ARCH_DEEPSEEK41;
            p.hparams.n_embd      = dim;
            p.hparams.n_layer_all = 1;
            p.hparams.n_head_arr.fill(1);
            p.hparams.n_head_kv_arr.fill(1);
            p.hparams.n_expert = experts;
            p.hparams.n_expert_used_arr.fill(k);
            p.hparams.n_ff_arr.fill(ff);
            p.hparams.swiglu_clamp_exp[0] = 10.0f;
            p.cparams.n_ctx               = 512;
            p.ubatch.n_tokens             = n;
            p.n_outputs                   = n;
            p.loras                       = &loras;
            p.sched                       = sched.get();
            p.backend_cpu                 = cpu.get();
            p.res                         = &res;
            p.cb                          = [&](const llama_ubatch &, ggml_tensor * t, const char * name, int) {
                ggml_set_name(t, name);
                if (mode == "cpu_lane" && t->op == GGML_OP_MUL_MAT_ID &&
                    ggml_backend_buffer_is_host(t->src[0]->buffer)) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), t, cpu.get());
                }
            };
            llm_graph_context graph(p);
            auto *            ctx   = res.get_ctx();
            auto *            input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n);
            ggml_set_input(input);
            auto * probabilities = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, n);
            ggml_set_input(probabilities);
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, n);
            ggml_set_input(ids);
            for (auto * t : { input, probabilities, ids }) {
                ggml_backend_sched_set_tensor_backend(sched.get(), t, gpu.get());
            }
            ggml_tensor * w[3];
            for (int i = 0; i < 3; ++i) {
                w[i] = mode == "cpu_lane" && (mask & (1 << i)) ? h[i] : g[i];
            }
            auto * output = graph.build_moe_ffn(input, nullptr, w[0], w[1], w[2], nullptr, experts, k, LLM_FFN_SILU,
                                                true, 1.5f, LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS, 0,
                                                probabilities, nullptr, nullptr, nullptr, nullptr, ids);
            ggml_set_output(output);
            ggml_build_forward_expand(res.get_gf(), output);
            ggml_backend_sched_set_tensor_backend(sched.get(), output, gpu.get());
            require(ggml_backend_sched_alloc_graph(sched.get(), res.get_gf()), "graph allocation failed");
            json placements = json::array();
            for (int j = 0; j < ggml_graph_n_nodes(res.get_gf()); ++j) {
                auto * node = ggml_graph_node(res.get_gf(), j);
                if (node->op != GGML_OP_MUL_MAT_ID) {
                    continue;
                }
                auto * assigned = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                require(assigned != nullptr, "unassigned expert projection");
                bool host_weight = ggml_backend_buffer_is_host(node->src[0]->buffer);
                require(assigned == (host_weight ? cpu.get() : gpu.get()), "expert projection backend mismatch");
                placements.push_back({
                    { "weight",  node->src[0]->name          },
                    { "backend", ggml_backend_name(assigned) }
                });
            }
            require(placements.size() == 3, "expected three expert projections");
            std::vector<float>                    x(dim * n), probs(experts * n, 0.0f), values(dim * n);
            std::mt19937                          rng(42);
            std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
            for (float & v : x) {
                v = dist(rng);
            }
            ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(probabilities, probs.data(), 0, probs.size() * sizeof(float));
            std::vector<double>  samples;
            std::vector<int32_t> chosen(k * n);
            size_t               transferred = 0;
            double               max_error = 0, max_output = 0, error_l2 = 0, reference_l2 = 0;
            for (int r = 0; r < count; ++r) {
                std::set<int> unique;
                for (int t = 0; t < n; ++t) {
                    for (int j = 0; j < k; ++j) {
                        chosen[t * k + j] = (r * 47 + t * k + j) % experts;
                        unique.insert(chosen[t * k + j]);
                    }
                }
                ggml_backend_tensor_set(ids, chosen.data(), 0, chosen.size() * sizeof(int32_t));
                auto   begin = clock_type::now();
                size_t bytes = 0;
                if (mode == "serial_h2d_gpu") {
                    for (int i = 0; i < 3; ++i) {
                        if (mask & (1 << i)) {
                            for (int e : unique) {
                                ggml_backend_tensor_set(g[i], static_cast<const char *>(h[i]->data) + e * h[i]->nb[2],
                                                        e * g[i]->nb[2], h[i]->nb[2]);
                                bytes += h[i]->nb[2];
                            }
                        }
                    }
                }
                require(ggml_backend_sched_graph_compute(sched.get(), res.get_gf()) == GGML_STATUS_SUCCESS,
                        "graph compute failed");
                ggml_backend_sched_synchronize(sched.get());
                double us = elapsed(begin);
                ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
                for (size_t j = 0; j < values.size(); ++j) {
                    require(std::isfinite(values[j]), "nonfinite output");
                    max_output   = std::max(max_output, std::abs(double(values[j])));
                    double ref   = mode == "gpu_resident" ? values[j] : reference[r][j];
                    double error = double(values[j]) - ref;
                    max_error    = std::max(max_error, std::abs(error));
                    error_l2 += error * error;
                    reference_l2 += ref * ref;
                }
                if (mode == "gpu_resident") {
                    reference.push_back(values);
                }
                if (r >= int(config["warmup"])) {
                    samples.push_back(us);
                    transferred += bytes;
                }
            }
            if (mode == "serial_h2d_gpu") {
                require(max_error < 0.01, "resident/streamed output mismatch");
            }
            result["modes"][mode]                            = stats(samples);
            result["modes"][mode]["projection_backends"]     = placements;
            result["modes"][mode]["max_abs_vs_gpu"]          = max_error;
            result["modes"][mode]["output_max_abs"]          = max_output;
            result["modes"][mode]["relative_l2_vs_gpu"]      = std::sqrt(error_l2 / std::max(reference_l2, 1e-30));
            result["modes"][mode]["transfer_bytes_per_call"] = transferred / int(config["repeats"]);
        }
        return result;
    }

    json storage() {
        std::vector<double>  ram, direct;
        std::vector<uint8_t> scratch;
        size_t               bytes = 0;
        for (int i = 0; i < 3; ++i) {
            if (mask & (1 << i)) {
                bytes += h[i]->nb[2];
            }
        }
        scratch.resize(bytes);
        uint64_t checksum = 0;
        int      error    = 0;
        for (int r = 0; r < int(config["repeats"]); ++r) {
            int    e      = (r * 47) % experts;
            size_t cursor = 0;
            auto   start  = clock_type::now();
            for (int i = 0; i < 3; ++i) {
                if (mask & (1 << i)) {
                    memcpy(scratch.data() + cursor, static_cast<const char *>(h[i]->data) + e * h[i]->nb[2],
                           h[i]->nb[2]);
                    cursor += h[i]->nb[2];
                }
            }
            ram.push_back(elapsed(start));
            for (auto v : scratch) {
                checksum += v;
            }
            double us = 0;
            for (int i = 0; i < 3 && error == 0; ++i) {
                if (mask & (1 << i)) {
                    const auto &      desc = config["tensors"][i];
                    const std::string path = desc["path"];
                    int               fd   = ::open(path.c_str(), O_RDONLY | O_DIRECT);
                    if (fd < 0) {
                        error = errno;
                        break;
                    }
                    size_t offset = desc["offset"].get<size_t>() + e * h[i]->nb[2], aligned = offset / 4096 * 4096;
                    size_t length = (offset - aligned + h[i]->nb[2] + 4095) / 4096 * 4096;
                    void * memory = nullptr;
                    if (posix_memalign(&memory, 4096, length) != 0) {
                        close(fd);
                        error = ENOMEM;
                        break;
                    }
                    auto    begin = clock_type::now();
                    ssize_t got   = pread(fd, memory, length, aligned);
                    us += elapsed(begin);
                    if (got < 0 || size_t(got) < offset - aligned + h[i]->nb[2]) {
                        error = got < 0 ? errno : EIO;
                    } else {
                        require(memcmp(static_cast<char *>(memory) + offset - aligned,
                                       static_cast<const char *>(h[i]->data) + e * h[i]->nb[2], h[i]->nb[2]) == 0,
                                "direct read differs from mapped source");
                    }
                    free(memory);
                    close(fd);
                }
            }
            if (!error) {
                direct.push_back(us);
            }
        }
        json result = {
            { "bytes_per_expert_host_bundle", bytes      },
            { "warm_mmap_memcpy",             stats(ram) },
            { "checksum_sink",                checksum   },
            { "direct_io_supported",          error == 0 },
            { "direct_io_errno",              error      }
        };
        if (!error) {
            result["direct_storage_read"] = stats(direct);
        }
        return result;
    }

    ~bench() {
        sched.reset();
        weights.reset();
        device.reset();
        host.reset();
        gpu.reset();
        cpu.reset();
    }
};

int main(int argc, char ** argv) {
    json result = {
        { "complete", false         },
        { "cases",    json::array() }
    };
    try {
        require(argc == 3, "usage: llama-moe-cache-calibrate CONFIG OUTPUT");
        std::ifstream input(argv[1]);
        require(bool(input), "cannot open config");
        json  cfg = json::parse(input);
        bench test;
        test.init(cfg);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(ggml_backend_get_device(test.gpu.get()), &props);
        result["device"] = {
            { "name",        props.name                                    },
            { "description", props.description                             },
            { "pci_bus_id",  props.device_id ? props.device_id : "unknown" }
        };
        result["initial_upload_bytes"] = test.config["initial_upload_bytes"];
        for (const auto & c : cfg.at("cases")) {
            auto row = test.measure(c["tokens"], c["picks"]);
            result["cases"].push_back(row);
            save(argv[2], result);
        }
        result["storage"]  = test.storage();
        result["complete"] = true;
        save(argv[2], result);
        return 0;
    } catch (const std::exception & error) {
        result["error"] = error.what();
        if (argc == 3) {
            save(argv[2], result);
        }
        fprintf(stderr, "calibration failed: %s\n", error.what());
        return 1;
    }
}
