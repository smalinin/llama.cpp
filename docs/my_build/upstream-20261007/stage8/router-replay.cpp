#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    require(argc == 3, "usage: router-replay WEIGHT_BF16 INPUT_F32");
    const int k = 5120, m = 384;
    std::vector<uint16_t> raw(k*m);
    std::vector<float> weights(k*m), input(k);
    std::ifstream a(argv[1], std::ios::binary), b(argv[2], std::ios::binary);
    require(bool(a.read(reinterpret_cast<char *>(raw.data()), raw.size()*2)), "weight read failed");
    require(bool(b.read(reinterpret_cast<char *>(input.data()), input.size()*4)), "input read failed");
    for (size_t i = 0; i < raw.size(); ++i) {
        const uint32_t bits = uint32_t(raw[i]) << 16;
        std::memcpy(&weights[i], &bits, 4);
    }
    std::vector<double> reference(m);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) reference[row] += double(weights[row*k+col])*input[col];
    }
    auto backend = ggml_backend_cuda_init(0);
    require(backend, "CUDA backend failed");
    for (auto type : {GGML_TYPE_BF16, GGML_TYPE_F32}) {
        for (auto prec : {GGML_PREC_DEFAULT, GGML_PREC_F32}) {
            std::vector<float> scalar;
            for (int width : {1,2,3,4,5,16}) {
                ggml_init_params params{16*1024*1024, nullptr, true};
                auto * ctx = ggml_init(params);
                auto * w = ggml_new_tensor_2d(ctx, type, k, m);
                auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, width);
                auto * y = ggml_mul_mat(ctx, w, x);
                ggml_mul_mat_set_prec(y, prec);
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, y);
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                require(buffer, "allocation failed");
                if (type == GGML_TYPE_BF16) ggml_backend_tensor_set(w, raw.data(), 0, raw.size()*2);
                else ggml_backend_tensor_set(w, weights.data(), 0, weights.size()*4);
                for (int col = 0; col < width; ++col) ggml_backend_tensor_set(x, input.data(), col*k*4, k*4);
                require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph failed");
                std::vector<float> result(m*width);
                ggml_backend_tensor_get(y, result.data(), 0, result.size()*4);
                if (width == 1) scalar = result;
                double err=0, maxerr=0, scalarerr=0, scalarrms=0, colerr=0;
                for (int row = 0; row < m; ++row) {
                    const double d = result[row]-reference[row];
                    err += d*d;
                    maxerr = std::max(maxerr, std::abs(d));
                    scalarerr = std::max(scalarerr, std::abs(double(result[row])-scalar[row]));
                    scalarrms += std::pow(double(result[row])-scalar[row], 2);
                    for (int col=1; col<width; ++col) colerr=std::max(colerr, std::abs(double(result[row])-result[col*m+row]));
                }
                std::printf("{\"type\":\"%s\",\"prec\":%d,\"width\":%d,\"cpu_max_abs\":%.12g,\"cpu_rms\":%.12g,\"scalar_max_abs\":%.12g,\"scalar_rms\":%.12g,\"column_max_abs\":%.12g}\n",
                    ggml_type_name(type), int(prec), width, maxerr, std::sqrt(err/m), scalarerr, std::sqrt(scalarrms/m), colerr);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(backend);
}
