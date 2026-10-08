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
        check(argc == 3, "usage: ffn-replay INPUT OUTPUT");
        const std::string root = argv[1], output = argv[2];
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
        for (int source : {1, 2, 4}) {
            const auto input = read<float>(root+"/input-w"+std::to_string(source)+".f32", 5120*source);
            const auto captured = read<float>(root+"/captured-w"+std::to_string(source)+".f32", 5120*source);
            std::vector<float> scalar;
            for (int width : {1, 2, 4}) {
                for (bool actual : {false, true}) {
                    if (actual && (width != source || source == 1)) continue;
                    const std::string label = "inputw"+std::to_string(source)+"-batch"+std::to_string(width)+(actual ? "-actual" : "-repeat");
                    auto ctx = ggml_init({4*1024*1024, nullptr, true});
                    auto graph = ggml_new_graph(ctx);
                    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, width);
                    auto logits = ggml_mul_mat(ctx, w("ffn_gate_inp.weight"), x);
                    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
                    auto probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
                    auto selection = ggml_add(ctx, probs, w("exp_probs_b.bias"));
                    auto ids = ggml_argsort_top_k(ctx, selection, 6);
                    auto probabilities = ggml_reshape_3d(ctx, probs, 1, 384, width);
                    auto expert_weights = ggml_get_rows(ctx, probabilities, ids);
                    expert_weights = ggml_reshape_2d(ctx, expert_weights, 6, width);
                    auto sum = ggml_clamp(ctx, ggml_sum_rows(ctx, expert_weights), 6.103515625e-5f, INFINITY);
                    expert_weights = ggml_div(ctx, expert_weights, sum);
                    expert_weights = ggml_scale(ctx, ggml_reshape_3d(ctx, expert_weights, 1, 6, width), 1.5f);
                    ggml_build_forward_expand(graph, expert_weights);
                    auto mx = ggml_reshape_3d(ctx, x, 5120, 1, width);
                    auto up = ggml_mul_mat_id(ctx, w("ffn_up_exps.weight"), mx, ids);
                    auto gate = ggml_mul_mat_id(ctx, w("ffn_gate_exps.weight"), mx, ids);
                    auto hidden = ggml_swiglu_clamp(ctx, gate, up, 10.0f);
                    auto down = ggml_mul_mat_id(ctx, w("ffn_down_exps.weight"), hidden, ids);
                    auto weighted = ggml_mul(ctx, down, expert_weights);
                    ggml_build_forward_expand(graph, weighted);
                    std::vector<ggml_tensor *> views;
                    for (int i = 0; i < 6; ++i) {
                        auto view = ggml_view_2d(ctx, weighted, 5120, width, weighted->nb[2], i*weighted->nb[1]);
                        ggml_build_forward_expand(graph, view);
                        views.push_back(view);
                    }
                    auto moe = views[0];
                    for (int i = 1; i < 6; ++i) {
                        moe = ggml_add(ctx, moe, views[i]);
                        ggml_build_forward_expand(graph, moe);
                    }
                    auto shared_up = ggml_mul_mat(ctx, w("ffn_up_shexp.weight"), x);
                    auto shared_gate = ggml_mul_mat(ctx, w("ffn_gate_shexp.weight"), x);
                    auto shared_hidden = ggml_swiglu_clamp(ctx, shared_gate, shared_up, 10.0f);
                    auto shared = ggml_mul_mat(ctx, w("ffn_down_shexp.weight"), shared_hidden);
                    auto result = ggml_add(ctx, moe, shared);
                    ggml_build_forward_expand(graph, result);
                    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    check(buffer, "graph allocation failed");
                    for (int col = 0; col < width; ++col) ggml_backend_tensor_set(x, input.data()+(actual ? col*5120 : 0), col*5120*4, 5120*4);
                    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "FFN compute failed");
                    auto values = save(output+"/"+label+"-out.f32", result);
                    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "FFN repeat failed");
                    std::vector<float> repeat(values.size());
                    ggml_backend_tensor_get(result, repeat.data(), 0, repeat.size()*4);
                    check(std::memcmp(values.data(), repeat.data(), values.size()*4) == 0, "FFN repeat differs");
                    save(output+"/"+label+"-moe.f32", moe);
                    save(output+"/"+label+"-moe-hidden.f32", hidden);
                    save(output+"/"+label+"-shared-hidden.f32", shared_hidden);
                    save(output+"/"+label+"-router.f32", logits);
                    save(output+"/"+label+"-weights.f32", expert_weights);
                    std::vector<int32_t> selected(6*width);
                    for (int col = 0; col < width; ++col) ggml_backend_tensor_get(ids, selected.data()+6*col, col*ids->nb[1], 6*4);
                    std::ofstream(output+"/"+label+"-ids.i32", std::ios::binary).write(reinterpret_cast<const char *>(selected.data()), selected.size()*4);
                    if (width == 1) scalar = values;
                    double maximum = 0, capture_max = 0, column_max = 0;
                    for (int i = 0; i < 5120; ++i) {
                        maximum = std::max(maximum, std::abs(double(values[i])-scalar[i]));
                        capture_max = std::max(capture_max, std::abs(double(values[i])-captured[i]));
                        if (!actual) for (int col = 1; col < width; ++col) column_max = std::max(column_max, std::abs(double(values[i])-values[col*5120+i]));
                    }
                    double all_capture_max = 0;
                    if (actual || source == 1) for (size_t i = 0; i < values.size(); ++i) all_capture_max = std::max(all_capture_max, std::abs(double(values[i])-captured[i%captured.size()]));
                    report << "{\"label\":\"" << label << "\",\"max_vs_scalar\":" << maximum << ",\"max_vs_capture_first\":" << capture_max
                           << ",\"max_vs_capture_all\":" << all_capture_max << ",\"column_max\":" << column_max << ",\"repeat_stable\":true,\"ids\":[";
                    for (size_t i = 0; i < selected.size(); ++i) report << (i ? "," : "") << selected[i];
                    report << "]}\n";
                    report.flush();
                    std::printf("DONE %s max_capture=%.9g\n", label.c_str(), capture_max);
                    ggml_backend_buffer_free(buffer);
                    ggml_free(ctx);
                }
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
