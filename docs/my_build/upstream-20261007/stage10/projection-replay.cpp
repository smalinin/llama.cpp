#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

template<typename T> static std::vector<T> read(const std::string & path, size_t count) {
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    check(bool(input.read(reinterpret_cast<char *>(values.data()), count*sizeof(T))), "cannot read input");
    return values;
}

static void write(const std::string & path, const std::vector<float> & values) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
}

int main(int argc, char ** argv) {
    check(argc == 6, "usage: projection-replay INPUT_DIR TYPE N M OUTPUT_DIR");
    const std::string root = argv[1], output = argv[5];
    const auto type = static_cast<ggml_type>(std::stoi(argv[2]));
    const int n = std::stoi(argv[3]), m = std::stoi(argv[4]);
    check(!std::filesystem::exists(output), "output exists");
    std::filesystem::create_directories(output);
    const auto weights = read<char>(root+"/weight.bin", ggml_row_size(type, n)*m);
    std::vector<float> dequantized(size_t(n)*m);
    const auto to_float = ggml_get_type_traits(type)->to_float;
    check(to_float, "missing dequantization");
    for (int row = 0; row < m; ++row) to_float(weights.data()+row*ggml_row_size(type, n), dequantized.data()+size_t(row)*n, n);
    auto backend = ggml_backend_cuda_init(0);
    check(backend, "CUDA backend failed");
    auto weights_ctx = ggml_init({1024*1024, nullptr, true});
    auto wq = ggml_new_tensor_2d(weights_ctx, type, n, m);
    auto wf = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, n, m);
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    check(weights_buffer, "weight allocation failed");
    ggml_backend_tensor_set(wq, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(wf, dequantized.data(), 0, dequantized.size()*4);
    std::ofstream report(output+"/results.jsonl");
    for (int input_width : {1, 2, 4}) {
        const auto input = read<float>(root+"/input-w"+std::to_string(input_width)+".f32", n);
        std::vector<float> reference(m);
        for (int row = 0; row < m; ++row) {
            double sum = 0;
            for (int j = 0; j < n; ++j) sum += double(dequantized[size_t(row)*n+j])*input[j];
            reference[row] = float(sum);
        }
        write(output+"/cpu-inputw"+std::to_string(input_width)+".f32", reference);
        for (bool fp32 : {false, true}) {
            std::vector<float> scalar;
            for (int width : {1, 2, 4}) {
                auto ctx = ggml_init({1024*1024, nullptr, true});
                auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, width);
                auto y = ggml_mul_mat(ctx, fp32 ? wf : wq, x);
                if (fp32) ggml_mul_mat_set_prec(y, GGML_PREC_F32);
                auto graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, y);
                check(ggml_graph_n_nodes(graph) == 1, "unexpected projection graph");
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                check(buffer, "projection allocation failed");
                for (int col = 0; col < width; ++col) ggml_backend_tensor_set(x, input.data(), size_t(col)*n*4, n*4);
                check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "projection failed");
                std::vector<float> actual(size_t(m)*width), repeat(actual.size());
                ggml_backend_tensor_get(y, actual.data(), 0, actual.size()*4);
                check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "projection repeat failed");
                ggml_backend_tensor_get(y, repeat.data(), 0, repeat.size()*4);
                check(std::memcmp(actual.data(), repeat.data(), actual.size()*4) == 0, "projection repeat differs");
                if (width == 1) scalar = actual;
                double max_delta = 0, column_delta = 0, cpu_delta = 0;
                for (int row = 0; row < m; ++row) {
                    max_delta = std::max(max_delta, std::abs(double(actual[row])-scalar[row]));
                    cpu_delta = std::max(cpu_delta, std::abs(double(actual[row])-reference[row]));
                    for (int col = 1; col < width; ++col) column_delta = std::max(column_delta, std::abs(double(actual[row])-actual[size_t(col)*m+row]));
                }
                const std::string label = std::string(fp32 ? "fp32" : "quant")+"-inputw"+std::to_string(input_width)+"-batch"+std::to_string(width);
                report << "{\"label\":\"" << label << "\",\"max_vs_scalar\":" << max_delta
                       << ",\"column_max\":" << column_delta << ",\"max_vs_cpu\":" << cpu_delta << ",\"repeat_stable\":true}\n";
                write(output+"/"+label+".f32", actual);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_ctx);
    ggml_backend_free(backend);
}
