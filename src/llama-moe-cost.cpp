#include "llama-moe-cost.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "llama-graph.h"
#include "ggml-cpp.h"
#include "../vendor/nlohmann/json.hpp"
extern "C" {
#include "../vendor/hash/sha256/sha256.h"
}
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <random>
#include <regex>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sched.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace {
using json = nlohmann::json;
void check(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
std::string text(const std::string & path) {
    std::ifstream f(path); check(bool(f), "cannot read calibration identity source");
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
std::string hex_hash(sha256_t & state) {
    std::array<unsigned char, SHA256_DIGEST_SIZE> bytes{};
    sha256_final(&state, bytes.data());
    std::ostringstream out;
    for (unsigned char b : bytes) { out << std::hex << std::setfill('0') << std::setw(2) << unsigned(b); }
    return out.str();
}
std::string hash(const void * ptr, size_t n) {
    sha256_t state; sha256_init(&state);
    sha256_update(&state, static_cast<const unsigned char *>(ptr), n);
    return hex_hash(state);
}
std::string file_hash(const std::string & path) {
    std::ifstream f(path, std::ios::binary); check(bool(f), "cannot read calibration library");
    sha256_t state; sha256_init(&state); std::array<unsigned char, 65536> buf{};
    while (f) { f.read(reinterpret_cast<char *>(buf.data()), buf.size()); sha256_update(&state, buf.data(), f.gcount()); }
    check(f.eof(), "calibration library read error"); return hex_hash(state);
}
std::string trim(std::string value) {
    const auto a = value.find_first_not_of(" \t\r\n"), b = value.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : value.substr(a, b-a+1);
}
#ifdef __linux__
struct measurement {
    double median_us = 0, min_us = 0, max_us = 0, relative_l2 = 0, max_abs = 0, magnitude = 0;
    std::vector<std::vector<float>> output;
    std::vector<double> samples;
};
measurement benchmark(const llama_model & model, const llama_layer & layer, ggml_backend_dev_t origin,
        ggml_backend_dev_t destination, int threads, bool staged, int repeats = 3, int warmup = 2) {
    const int n = 512, k = model.hparams.n_expert_used(), dim = layer.ffn_up_exps->ne[0];
    const int ff = layer.ffn_up_exps->ne[1], experts = layer.ffn_up_exps->ne[2];
    check(k > 0 && k <= 6 && threads > 0, "unsupported startup calibration geometry");
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    check(bool(cpu), "cannot initialize calibration CPU");
    auto * cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu.get()));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    check(set_threads, "calibration CPU backend has no thread-count interface");
    set_threads(cpu.get(), threads);
    std::vector<ggml_backend_ptr> gpu;
    std::vector<ggml_backend_t> backends;
    ggml_backend_t home_backend = nullptr, stage_backend = nullptr;
    for (const auto & device : model.devices) {
        if (ggml_backend_dev_type(device.dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device.dev)), "CUDA") != 0) { continue; }
        gpu.emplace_back(ggml_backend_dev_init(device.dev, nullptr));
        check(bool(gpu.back()), "cannot initialize calibration GPU");
        backends.push_back(gpu.back().get());
        if (device.dev == origin) { home_backend = backends.back(); }
        if (device.dev == destination) { stage_backend = backends.back(); }
    }
    check(home_backend && (!staged || stage_backend), "calibration GPU is not active");
    backends.push_back(cpu.get());
    ggml_context_ptr pools_ctx(ggml_init({1024*1024, nullptr, true}));
    ggml_context_ptr pinned_ctx(ggml_init({1024*1024, nullptr, true}));
    ggml_backend_buffer_ptr pool_buffer, pinned_buffer;
    ggml_tensor * pools[5]{};
    if (staged) {
        size_t expert_bytes = 0;
        for (auto * t : {layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps}) {
            if (ggml_backend_buffer_is_host(t->buffer) && t->type == GGML_TYPE_Q3_K) { expert_bytes = std::max(expert_bytes, t->nb[2]); }
        }
        check(expert_bytes != 0, "no supported host Q3_K projection");
        const size_t routes = size_t(n)*k, weight_bytes = 2*(4*expert_bytes + 512), work = std::max(dim, ff)*routes;
        pools[0] = ggml_new_tensor_1d(pools_ctx.get(), GGML_TYPE_I8, weight_bytes);
        pools[1] = ggml_new_tensor_1d(pools_ctx.get(), GGML_TYPE_F32, work);
        pools[2] = ggml_new_tensor_1d(pools_ctx.get(), GGML_TYPE_F32, work);
        pools[3] = ggml_new_tensor_1d(pools_ctx.get(), GGML_TYPE_I32, 6*routes);
        pools[4] = ggml_new_tensor_1d(pinned_ctx.get(), GGML_TYPE_I8, weight_bytes+6*routes*sizeof(int32_t));
        pool_buffer.reset(ggml_backend_alloc_ctx_tensors(pools_ctx.get(), stage_backend));
        auto * host_type = ggml_backend_dev_host_buffer_type(destination);
        check(host_type, "pinned host memory unavailable");
        pinned_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(pinned_ctx.get(), host_type));
        check(pool_buffer && pinned_buffer && ggml_backend_buffer_get_type(pinned_buffer.get()) == host_type, "calibration workspace allocation failed");
    }
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 8192, false, true));
    check(bool(sched), "calibration scheduler allocation failed");
    llm_graph_result res(8192); res.reset();
    llama_adapter_loras loras;
    llm_graph_params p{}; p.arch = model.arch; p.hparams = model.hparams;
    p.cparams.n_ctx = 512; p.ubatch.n_tokens = n; p.n_outputs = n; p.loras = &loras;
    p.sched = sched.get(); p.backend_cpu = cpu.get(); p.res = &res; p.disable_moe_cache = true;
    p.cb = [&](const llama_ubatch &, ggml_tensor * t, const char * name, int) {
        ggml_set_name(t, name);
        if (t->op != GGML_OP_MUL_MAT_ID || !ggml_backend_buffer_is_host(t->src[0]->buffer)) { return; }
        if (staged && t->src[0]->type == GGML_TYPE_Q3_K) {
            t->op = GGML_OP_MUL_MAT_ID_STAGED;
            for (int i = 0; i < 5; ++i) { t->src[i+3] = pools[i]; }
            const int wave = 4; std::memcpy(t->op_params, &wave, sizeof(wave));
            ggml_backend_sched_set_tensor_backend(sched.get(), t, stage_backend);
        } else { ggml_backend_sched_set_tensor_backend(sched.get(), t, cpu.get()); }
    };
    llm_graph_context graph(p); auto * ctx = res.get_ctx();
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, n);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, n);
    for (auto * t : {input, logits, ids}) { ggml_set_input(t); ggml_backend_sched_set_tensor_backend(sched.get(), t, home_backend); }
    ggml_tensor * weights[3]; int i = 0;
    for (auto * t : {layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps}) {
        auto * alias = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], t->ne[2]);
        alias->buffer = t->buffer; alias->data = t->data; weights[i++] = alias;
    }
    auto * output = graph.build_moe_ffn(input, nullptr, weights[0], weights[1], weights[2], nullptr,
            experts, k, LLM_FFN_SILU, model.hparams.expert_weights_norm, model.hparams.expert_weights_scale,
            (llama_expert_gating_func_type) model.hparams.expert_gating_func, int(&layer-model.layers.data()),
            logits, nullptr, nullptr, nullptr, nullptr, ids);
    ggml_set_output(output); ggml_build_forward_expand(res.get_gf(), output);
    ggml_backend_sched_set_tensor_backend(sched.get(), output, home_backend);
    check(ggml_backend_sched_alloc_graph(sched.get(), res.get_gf()), "calibration graph allocation failed");
    std::vector<float> x(dim*n), probs(experts*n, 0.0f), values(dim*n);
    std::mt19937 rng(42); std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for (auto & v : x) { v = distribution(rng); }
    std::vector<int32_t> chosen(k*n); measurement result;
    for (int r = 0; r < repeats+warmup; ++r) {
        for (int t = 0; t < n; ++t) { for (int j = 0; j < k; ++j) { chosen[t*k+j] = (r*47+t*k+j)%experts; } }
        ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
        ggml_backend_tensor_set(logits, probs.data(), 0, probs.size()*sizeof(float));
        ggml_backend_tensor_set(ids, chosen.data(), 0, chosen.size()*sizeof(int32_t));
        const auto start = std::chrono::steady_clock::now();
        check(ggml_backend_sched_graph_compute(sched.get(), res.get_gf()) == GGML_STATUS_SUCCESS, "calibration graph compute failed");
        ggml_backend_sched_synchronize(sched.get());
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count();
        ggml_backend_tensor_get(output, values.data(), 0, values.size()*sizeof(float));
        for (float v : values) { check(std::isfinite(v), "nonfinite startup calibration output"); result.magnitude = std::max(result.magnitude, std::abs(double(v))); }
        if (r >= warmup) { result.samples.push_back(us); result.output.push_back(values); }
    }
    auto sorted = result.samples; std::sort(sorted.begin(), sorted.end());
    result.min_us = sorted.front(); result.max_us = sorted.back(); result.median_us = sorted[sorted.size()/2];
    sched.reset(); return result;
}
json cost_record(const measurement & value) {
    return {{"median_us", value.median_us}, {"min_us", value.min_us}, {"max_us", value.max_us},
            {"samples_us", value.samples}, {"relative_l2", value.relative_l2}, {"max_abs", value.max_abs}, {"magnitude", value.magnitude}};
}
void compare(measurement & candidate, const measurement & reference) {
    check(candidate.output.size() == reference.output.size(), "calibration output count mismatch");
    double error2 = 0, energy = 0;
    for (size_t r = 0; r < reference.output.size(); ++r) {
        check(candidate.output[r].size() == reference.output[r].size(), "calibration output geometry mismatch");
        for (size_t i = 0; i < reference.output[r].size(); ++i) {
            const double a = reference.output[r][i], d = candidate.output[r][i] - a;
            energy += a*a; error2 += d*d; candidate.max_abs = std::max(candidate.max_abs, std::abs(d));
        }
    }
    candidate.relative_l2 = std::sqrt(error2/std::max(energy, 1e-30));
    check(candidate.relative_l2 <= 0.03 && candidate.max_abs <= 0.1*std::max(reference.magnitude, 1e-6), "startup FFN numerical tolerance exceeded");
}
json startup_identity(const llama_model & model, ggml_backend_dev_t home,
        const std::vector<const llama_layer *> & layers, int threads) {
    json layout = json::array(), devices = json::array();
    for (const auto * l : layers) {
        json row = {{"layer", l-model.layers.data()}, {"swiglu_clamp", model.hparams.swiglu_clamp_exp[l-model.layers.data()]}, {"tensors", json::array()}};
        for (const auto * t : {l->ffn_up_exps, l->ffn_gate_exps, l->ffn_down_exps}) {
            json samples = json::array(); std::array<unsigned char, 128> bytes{};
            for (size_t offset : {size_t(0), ggml_nbytes(t)-bytes.size()}) {
                ggml_backend_tensor_get(t, bytes.data(), offset, bytes.size()); samples.push_back(hash(bytes.data(), bytes.size()));
            }
            row["tensors"].push_back({{"name", t->name}, {"type", t->type},
                {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
                {"host", ggml_backend_buffer_is_host(t->buffer)}, {"buffer", ggml_backend_buffer_name(t->buffer)}, {"samples", samples}});
        }
        layout.push_back(row);
    }
    for (const auto & device : model.devices) {
        ggml_backend_dev_props props{}; ggml_backend_dev_get_props(device.dev, &props);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device.dev)), "CUDA") != 0) { continue; }
        check(props.device_id, "GPU has no PCI identity"); const std::string id = props.device_id;
        devices.push_back({{"pci", id}, {"name", props.description}, {"memory", props.memory_total},
                {"information", text("/proc/driver/nvidia/gpus/"+id+"/information")},
                {"pcie_width", trim(text("/sys/bus/pci/devices/"+id+"/current_link_width"))},
                {"pcie_max_speed", trim(text("/sys/bus/pci/devices/"+id+"/max_link_speed"))}});
    }
    std::set<std::string> paths; std::istringstream maps(text("/proc/self/maps")); std::string line;
    while (std::getline(maps, line)) {
        const auto pos = line.find(" /"); if (pos == std::string::npos) { continue; }
        const std::string path = line.substr(pos+1), name = std::filesystem::path(path).filename().string();
        if (name.find("libllama") == 0 || name.find("libggml") == 0) { paths.insert(path); }
    }
    check(paths.size() >= 4, "cannot identify loaded calibration libraries"); json runtime = json::array();
    for (const auto & path : paths) { runtime.push_back({{"path", path}, {"sha256", file_hash(path)}}); }
    cpu_set_t affinity; check(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "cannot read CPU affinity");
    std::vector<int> cpus; for (int i = 0; i < CPU_SETSIZE; ++i) { if (CPU_ISSET(i, &affinity)) { cpus.push_back(i); } }
    std::string cpuinfo; std::istringstream cpu(text("/proc/cpuinfo"));
    while (std::getline(cpu, line) && !line.empty()) {
        if (line.find("model name") == 0 || line.find("flags") == 0 || line.find("microcode") == 0) { cpuinfo += line+'\n'; }
    }
    std::string memory_nodes; std::istringstream status(text("/proc/self/status"));
    while (std::getline(status, line)) { if (line.find("Mems_allowed_list:") == 0) { memory_nodes = line; } }
    json environment = json::object();
    for (const char * name : {"GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS", "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_MMVF_SAMPLE_BATCH", "GGML_CUDA_MOE_DOWN_REDUCE_EXTENDED", "GGML_CUDA_GRAPH_OPT", "GGML_CUDA_PROJECTION_FAN"}) {
        const char * value = std::getenv(name); environment[name] = value ? json(value) : json(nullptr);
    }
    ggml_backend_dev_props origin{}; ggml_backend_dev_get_props(home, &origin);
    json sources = json::array();
    if (!model.source_path.empty()) {
        std::filesystem::path first = model.source_path;
        std::smatch match; const auto name = first.filename().string();
        int count = 1; std::string prefix;
        if (std::regex_match(name, match, std::regex("(.*)-00001-of-([0-9]{5})[.]gguf"))) {
            count = std::stoi(match[2]); prefix = match[1]; check(count > 0 && count <= 10000, "unsupported shard count");
        }
        for (int i = 1; i <= count; ++i) {
            auto path = first;
            if (!prefix.empty()) {
                std::ostringstream part; part << prefix << '-' << std::setfill('0') << std::setw(5) << i << "-of-" << std::setw(5) << count << ".gguf";
                path = first.parent_path()/part.str();
            }
            struct stat st{}; check(stat(path.c_str(), &st) == 0, "cannot stat calibrated model source");
            sources.push_back({{"path", path.string()}, {"device", st.st_dev}, {"inode", st.st_ino}, {"bytes", st.st_size},
                    {"mtime_ns", uint64_t(st.st_mtim.tv_sec)*1000000000 + st.st_mtim.tv_nsec}});
        }
    }
    std::map<std::string, std::string> metadata(model.gguf_kv.begin(), model.gguf_kv.end());
    return {{"revision", 1}, {"sources", sources}, {"model_path", model.source_path}, {"metadata", metadata}, {"layout", layout},
        {"runtime", runtime}, {"devices", devices}, {"home", origin.device_id ? origin.device_id : "unknown"},
        {"cpu", cpuinfo}, {"affinity", cpus}, {"memory_nodes", memory_nodes}, {"environment", environment}, {"numa", text("/sys/devices/system/node/online")},
        {"governor", text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")},
        {"driver", text("/proc/driver/nvidia/version")}, {"threads", threads},
        {"picks", model.hparams.n_expert_used()}, {"weights_norm", model.hparams.expert_weights_norm},
        {"weights_scale", model.hparams.expert_weights_scale}, {"gating_func", model.hparams.expert_gating_func},
        {"protocol", "512 tokens; uniform [-.25,.25], equal router logits; rotating distinct ids; reset every call;2 warmup+3 trials; actual staged operator; full FFN to home"}};
}
json read_document(const std::filesystem::path & path) {
    check(std::filesystem::file_size(path) <= 16*1024*1024, "startup cost file too large");
    check(!std::filesystem::is_symlink(path), "startup cost path is a symlink");
    const auto envelope = json::parse(text(path.string()));
    check(envelope.at("format") == "llama-moe-prefill-costs" && envelope.at("version") == 1, "foreign startup cost file");
    const auto payload = envelope.at("payload").get<std::string>();
    check(hash(payload.data(), payload.size()) == envelope.at("sha256"), "startup cost checksum mismatch");
    auto doc = json::parse(payload); check(doc.is_object(), "invalid startup cost document"); return doc;
}
void save_document(const std::filesystem::path & path, const std::filesystem::path & lock, const json & document) {
    const auto payload = document.dump();
    const json envelope = {{"format", "llama-moe-prefill-costs"}, {"version", 1}, {"payload", payload}, {"sha256", hash(payload.data(), payload.size())}};
    const auto temp = lock/"costs.tmp";
    { std::ofstream file(temp); file << envelope.dump(2) << '\n'; file.flush(); check(bool(file), "cannot write startup costs"); }
    const int fd = open(temp.c_str(), O_RDONLY); check(fd >= 0, "cannot reopen startup costs");
    const int result = fsync(fd); close(fd); check(result == 0, "cannot sync startup costs");
    std::filesystem::rename(temp, path);
}
struct cost_lock {
    std::filesystem::path path; bool held = false;
    ~cost_lock() { if (held) { std::error_code e; std::filesystem::remove(path/"costs.tmp", e); std::filesystem::remove(path, e); } }
};
#endif
} // namespace

ggml_backend_buffer_type_t llama_moe_stage_auto(const llama_model & model,
        ggml_backend_buffer_type_t home, const std::vector<const llama_layer *> & layers,
        int32_t threads, int32_t max_tokens, size_t cache_bytes,
        std::map<ggml_backend_dev_t, size_t> & reserved) {
#ifndef __linux__
    LLAMA_LOG_INFO("moe-cache: auto CPU fallback: startup calibration requires Linux\n"); return nullptr;
#else
    try {
        check(model.arch == LLM_ARCH_DEEPSEEK41 && max_tokens == 512, "auto calibration supports DeepSeek41 chunk512");
        if (model.hparams.no_alloc) { return home; }
        auto * origin = ggml_backend_buft_get_device(home);
        const auto identity = startup_identity(model, origin, layers, threads);
        const auto key_text = identity.dump(), key = hash(key_text.data(), key_text.size());
        std::filesystem::path file;
        if (const char * path = std::getenv("LLAMA_MOE_PREFILL_STAGE_COSTS"); path && path[0]) { file = path; }
        else if (!model.source_path.empty()) { file = model.source_path + ".moe-prefill-costs.json"; }
        json document = json::object(), records = json::array();
        cost_lock lock; bool writable = !file.empty();
        if (writable) {
            lock.path = file.string()+".lock";
            try {
                lock.held = std::filesystem::create_directory(lock.path);
                if (!lock.held) { LLAMA_LOG_INFO("moe-cache: auto CPU fallback: startup cost writer lock exists; saved costs preserved\n"); return nullptr; }
            }
            catch (const std::exception & e) { writable = false; LLAMA_LOG_WARN("moe-cache: costs remain in memory: %s\n", e.what()); }
            if (std::filesystem::exists(file)) {
                try { document = read_document(file); }
                catch (const std::exception & e) { writable = false; LLAMA_LOG_WARN("moe-cache: preserving cost file, recalibrating in memory: %s\n", e.what()); }
            }
        }
        bool reuse = document.contains(key);
        if (reuse) { records = document.at(key).at("records"); check(document.at(key).at("identity") == identity, "startup identity hash collision"); }
        size_t expert_bytes = 0, dimension = 0;
        for (auto * l : layers) { for (auto * t : {l->ffn_up_exps, l->ffn_gate_exps, l->ffn_down_exps}) {
            if (ggml_backend_buffer_is_host(t->buffer) && t->type == GGML_TYPE_Q3_K) {
                expert_bytes = std::max(expert_bytes, t->nb[2]); dimension = std::max(dimension, size_t(std::max(t->ne[0], t->ne[1])));
            }
        } }
        const size_t routes = 512*model.hparams.n_expert_used_max();
        const size_t scratch = 2*(4*expert_bytes+512) + 2*dimension*routes*sizeof(float) + 6*routes*sizeof(int32_t) + 4096;
        auto available = [&](ggml_backend_dev_t dev) {
            if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) { return false; }
            size_t free = 0, total = 0; ggml_backend_dev_memory(dev, &free, &total);
            size_t required = reserved[dev];
            for (size_t bytes : {scratch, size_t(1024)*1024*1024, dev == origin ? cache_bytes : size_t(0)}) {
                if (bytes > std::numeric_limits<size_t>::max()-required) { return false; }
                required += bytes;
            }
            return expert_bytes != 0 && free >= required;
        };
        if (std::none_of(model.devices.begin(), model.devices.end(), [&](const llama_device & d) { return available(d.dev); })) {
            LLAMA_LOG_INFO("moe-cache: auto CPU fallback: no staging GPU has workspace/cache headroom plus 1024 MiB reserve; saved costs preserved\n");
            return nullptr;
        }
        if (reuse) {
            for (const auto & candidate : model.devices) {
                if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(candidate.dev)), "CUDA") != 0 || !available(candidate.dev)) { continue; }
                std::set<int> covered;
                for (const auto & row : records) { if (row.at("device") == ggml_backend_dev_name(candidate.dev)) { covered.insert(row.at("layer").get<int>()); } }
                if (covered.size() != layers.size()) { reuse = false; }
            }
        }
        std::string recheck_device;
        if (reuse) {
            double best_saved = std::numeric_limits<double>::infinity();
            for (const auto & candidate : model.devices) {
                if (!available(candidate.dev)) { continue; }
                double cost = 0; std::set<int> coverage;
                for (const auto & row : records) {
                    if (row.at("device") == ggml_backend_dev_name(candidate.dev)) { cost += row.at("stage").at("median_us").get<double>(); coverage.insert(row.at("layer").get<int>()); }
                }
                if (coverage.size() == layers.size() && cost < best_saved) { best_saved = cost; recheck_device = ggml_backend_dev_name(candidate.dev); }
            }
            if (recheck_device.empty()) { reuse = false; }
        }
        if (reuse) {
            try {
                for (const auto * l : layers) {
                    const int il = l-model.layers.data();
                    auto reference = benchmark(model, *l, origin, origin, threads, false, 3, 2);
                    bool checked = false;
                    for (auto & row : records) {
                        if (row.at("layer") != il || row.at("device") != recheck_device) { continue; }
                        auto * dev = ggml_backend_dev_by_name(row.at("device").get<std::string>().c_str());
                        check(dev && available(dev), "saved staging device is no longer available");
                        auto current = benchmark(model, *l, origin, dev, threads, true, 3, 2); compare(current, reference);
                        auto drift = [](double old, double now) { return std::abs(old-now) > std::max(20.0, old*0.25); };
                        LLAMA_LOG_INFO("moe-cache: recheck layer %d %s CPU %.3f -> %.3f ms, staged %.3f -> %.3f ms\n", il,
                                ggml_backend_dev_name(dev), row.at("cpu").at("median_us").get<double>()/1000, reference.median_us/1000,
                                row.at("stage").at("median_us").get<double>()/1000, current.median_us/1000);
                        check(!drift(row.at("cpu").at("median_us"), reference.median_us) &&
                              !drift(row.at("stage").at("median_us"), current.median_us), "startup cost recheck drift");
                        row["cpu"] = cost_record(reference); row["stage"] = cost_record(current); checked = true;
                    }
                    check(checked, "incomplete saved stage coverage");
                }
                LLAMA_LOG_INFO("moe-cache: startup costs reused after recheck: %s\n", file.c_str());
            } catch (const std::exception & e) { reuse = false; LLAMA_LOG_INFO("moe-cache: refreshing startup calibration: %s\n", e.what()); }
        }
        check(startup_identity(model, origin, layers, threads) == identity, "calibration identity changed during measurement");
        if (reuse) {
            document[key]["records"] = records;
            if (writable) { try { save_document(file, lock.path, document); } catch (const std::exception & e) { LLAMA_LOG_WARN("moe-cache: recheck costs remain in memory: %s\n", e.what()); } }
        }
        if (!reuse) {
            LLAMA_LOG_INFO("moe-cache: calibrating CPU and staged FFN, chunk512, warmup2+3 trials\n"); records = json::array();
            for (const auto * l : layers) {
                auto reference = benchmark(model, *l, origin, origin, threads, false);
                for (const auto & candidate : model.devices) {
                    if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(candidate.dev)), "CUDA") != 0 || !available(candidate.dev)) { continue; }
                    try {
                        auto value = benchmark(model, *l, origin, candidate.dev, threads, true); compare(value, reference);
                        records.push_back({{"layer", l-model.layers.data()}, {"device", ggml_backend_dev_name(candidate.dev)},
                                {"cpu", cost_record(reference)}, {"stage", cost_record(value)}});
                        LLAMA_LOG_INFO("moe-cache: calibration layer %td %s CPU %.3f ms, staged %.3f ms, relative L2 %.6f\n",
                                l-model.layers.data(), ggml_backend_dev_name(candidate.dev), reference.median_us/1000, value.median_us/1000, value.relative_l2);
                    } catch (const std::exception & e) { LLAMA_LOG_WARN("moe-cache: calibration rejects %s: %s\n", ggml_backend_dev_name(candidate.dev), e.what()); }
                }
            }
            check(startup_identity(model, origin, layers, threads) == identity, "calibration identity changed during measurement");
            document[key] = {{"identity", identity}, {"records", records}};
            if (writable) {
                try { save_document(file, lock.path, document); LLAMA_LOG_INFO("moe-cache: startup costs saved: %s\n", file.c_str()); }
                catch (const std::exception & e) { LLAMA_LOG_WARN("moe-cache: costs remain in memory: %s\n", e.what()); }
            }
        }
        double best = 0.95; ggml_backend_dev_t chosen = nullptr;
        for (const auto & candidate : model.devices) {
            if (!available(candidate.dev) || (reuse && recheck_device != ggml_backend_dev_name(candidate.dev))) { continue; }
            double cpu = 0, stage = 0; std::set<int> coverage;
            for (const auto & row : records) {
                if (row.at("device") != ggml_backend_dev_name(candidate.dev)) { continue; }
                const double a = row.at("cpu").at("median_us"), b = row.at("stage").at("median_us");
                if (b >= a*0.95) { coverage.clear(); break; }
                cpu += a; stage += b; coverage.insert(row.at("layer").get<int>());
            }
            if (coverage.size() == layers.size() && cpu > 0 && stage/cpu < best) { best = stage/cpu; chosen = candidate.dev; }
        }
        if (!chosen) { LLAMA_LOG_INFO("moe-cache: auto CPU fallback: no GPU measured >=5%% faster for chunk512\n"); return nullptr; }
        LLAMA_LOG_INFO("moe-cache: auto staging %s -> %s for chunk512, stage/CPU %.3f; shorter tails use CPU\n",
                ggml_backend_buft_name(home), ggml_backend_dev_name(chosen), best);
        reserved[chosen] += scratch;
        return ggml_backend_dev_buffer_type(chosen);
    } catch (const std::exception & e) { LLAMA_LOG_INFO("moe-cache: auto CPU fallback: %s\n", e.what()); return nullptr; }
#endif
}
