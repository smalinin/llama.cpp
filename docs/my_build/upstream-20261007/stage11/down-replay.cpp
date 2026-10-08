#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

template<typename T> static std::vector<T> read(const std::string & path, size_t count) {
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    check(bool(input.read(reinterpret_cast<char *>(values.data()), count*sizeof(T))), "input read failed");
    return values;
}

static std::vector<float> save(const std::string & path, ggml_tensor * tensor) {
    check(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "unsupported output tensor");
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*4);
    for (float value : values) check(std::isfinite(value), "nonfinite output");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(values.data()), values.size()*4);
    return values;
}

struct Weight {
    ggml_tensor * tensor;
    uint64_t offset, bytes;
    std::string path;
};

int main(int argc, char ** argv) {
    try {
        check(argc == 4, "usage: down-replay INPUT FFN OUTPUT");
        const std::string root = argv[1], ffn = argv[2], output = argv[3];
        check(!std::filesystem::exists(output), "output exists");
        std::filesystem::create_directories(output);
        auto backend = ggml_backend_cuda_init(0);
        check(backend, "CUDA initialization failed");
        auto weight_ctx = ggml_init({2*1024*1024, nullptr, true});
        std::map<std::string, Weight> weights;
        std::ifstream index(root+"/weight-index.tsv");
        for (std::string line; std::getline(index, line);) {
            std::istringstream input(line);
            std::string name, path;
            int type;
            int64_t ne[4];
            uint64_t offset, bytes;
            check(bool(input >> name >> type >> ne[0] >> ne[1] >> ne[2] >> ne[3] >> offset >> bytes >> path), "invalid tensor index");
            if (name != "blk.2.ffn_down_exps.weight") continue;
            auto tensor = ggml_new_tensor(weight_ctx, static_cast<ggml_type>(type), 4, ne);
            check(ggml_nbytes(tensor) == bytes, "tensor size mismatch");
            ggml_set_name(tensor, name.c_str());
            weights.emplace(name, Weight{tensor, offset, bytes, path});
        }
        auto weight_buffer = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
        check(weight_buffer, "weight allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<char> chunk(64*1024*1024);
        for (const auto & item : weights) {
            const auto & w = item.second;
            std::ifstream file(w.path, std::ios::binary);
            file.seekg(w.offset);
            for (uint64_t start = 0; start < w.bytes;) {
                const size_t count = std::min<uint64_t>(chunk.size(), w.bytes-start);
                check(bool(file.read(chunk.data(), count)), "weight read failed");
                ggml_backend_tensor_set(w.tensor, chunk.data(), start, count);
                start += count;
            }
            std::printf("LOADED %s %llu bytes\n", item.first.c_str(), (unsigned long long)w.bytes);
        }
        auto w = [&](const std::string & name) { return weights.at("blk.2."+name).tensor; };
        std::ofstream report(output+"/results.jsonl");
        const auto ids_data = read<int32_t>(ffn+"/inputw1-batch1-repeat-ids.i32", 6);
        const auto weights_data = read<float>(ffn+"/inputw1-batch1-repeat-weights.f32", 6);
        for (int source : {1, 2, 4}) {
            const auto input = read<float>(ffn+"/inputw1-batch"+std::to_string(source)+"-repeat-moe-hidden.f32", 2304*6);
            std::vector<float> scalar;
            for (int width : {1, 2, 4}) {
                const std::string label = "hiddenw"+std::to_string(source)+"-batch"+std::to_string(width);
                auto ctx = ggml_init({2*1024*1024, nullptr, true});
                auto graph = ggml_new_graph(ctx);
                auto hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2304, 6, width);
                auto all_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 384, width);
                auto ids = ggml_view_2d(ctx, all_ids, 6, width, all_ids->nb[1], 0);
                auto expert_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 6, width);
                auto down = ggml_mul_mat_id(ctx, w("ffn_down_exps.weight"), hidden, ids);
                auto weighted = ggml_mul(ctx, down, expert_weights);
                ggml_build_forward_expand(graph, weighted);
                std::vector<ggml_tensor *> views;
                for (int i = 0; i < 6; ++i) {
                    auto view = ggml_view_2d(ctx, weighted, 5120, width, weighted->nb[2], i*weighted->nb[1]);
                    ggml_build_forward_expand(graph, view);
                    views.push_back(view);
                }
                auto result = views[0];
                for (int i = 1; i < 6; ++i) {
                    result = ggml_add(ctx, result, views[i]);
                    ggml_build_forward_expand(graph, result);
                }
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                check(buffer, "graph allocation failed");
                for (int col = 0; col < width; ++col) {
                    ggml_backend_tensor_set(hidden, input.data(), col*input.size()*4, input.size()*4);
                    ggml_backend_tensor_set(all_ids, ids_data.data(), col*all_ids->nb[1], 6*4);
                    ggml_backend_tensor_set(expert_weights, weights_data.data(), col*6*4, 6*4);
                }
                check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "down compute failed");
                auto values = save(output+"/"+label+".f32", result);
                check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "down repeat failed");
                std::vector<float> repeat(values.size());
                ggml_backend_tensor_get(result, repeat.data(), 0, repeat.size()*4);
                check(std::memcmp(values.data(), repeat.data(), values.size()*4) == 0, "down repeat differs");
                if (width == 1) scalar = values;
                double maximum = 0, column_max = 0;
                for (int i = 0; i < 5120; ++i) {
                    maximum = std::max(maximum, std::abs(double(values[i])-scalar[i]));
                    for (int col = 1; col < width; ++col) column_max = std::max(column_max, std::abs(double(values[i])-values[col*5120+i]));
                }
                report << "{\"label\":\"" << label << "\",\"max_vs_scalar\":" << maximum
                       << ",\"column_max\":" << column_max << ",\"repeat_stable\":true}\n";
                report.flush();
                std::printf("DONE %s max_scalar=%.9g\n", label.c_str(), maximum);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weight_ctx);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
