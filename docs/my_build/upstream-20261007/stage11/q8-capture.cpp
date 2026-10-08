#include "ggml.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void quantize_row_q8_1_cuda(const float *, const int32_t *, void *, ggml_type,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
static void check(bool ok) { if (!ok) throw std::runtime_error("Q8 capture failed"); }
static void cuda_check(cudaError_t status) { check(status == cudaSuccess); }
int main(int argc, char ** argv) {
    check(argc == 4);
    const std::string input = argv[2], output = argv[3];
    check(!std::filesystem::exists(output));
    std::filesystem::create_directories(output);
    constexpr int n = 2304, padded = 2560, experts = 6;
    std::vector<float> x(n*experts);
    std::vector<char> quantized((padded/32)*36*experts), repeat(quantized.size());
    float * device_x = nullptr;
    void * device_q = nullptr;
    cuda_check(cudaSetDevice(0));
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_x), x.size()*4));
    cuda_check(cudaMalloc(&device_q, quantized.size()));
    for (int width : {1, 2, 4}) {
        std::ifstream f(input+"/inputw1-batch"+std::to_string(width)+"-repeat-moe-hidden.f32", std::ios::binary);
        check(bool(f.read(reinterpret_cast<char *>(x.data()), x.size()*4)));
        cuda_check(cudaMemcpy(device_x, x.data(), x.size()*4, cudaMemcpyHostToDevice));
        for (int attempt = 0; attempt < 2; ++attempt) {
            quantize_row_q8_1_cuda(device_x, nullptr, device_q, GGML_TYPE_Q3_K,
                                  n, n, n*experts, n*experts, padded, experts, 1, 1, nullptr);
            auto & target = attempt ? repeat : quantized;
            cuda_check(cudaMemcpy(target.data(), device_q, target.size(), cudaMemcpyDeviceToHost));
        }
        check(quantized == repeat);
        std::ofstream(output+"/hiddenw"+std::to_string(width)+".q8_1", std::ios::binary).write(quantized.data(), quantized.size());
    }
    cuda_check(cudaFree(device_q));
    cuda_check(cudaFree(device_x));
}
