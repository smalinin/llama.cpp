#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool ok) { if (!ok) throw std::runtime_error("CPU delta failed"); }
template<typename T> static std::vector<T> read(const std::string & path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    check(bool(f.read(reinterpret_cast<char *>(v.data()), n*sizeof(T))));
    return v;
}
static void save(const std::string & path, const std::vector<float> & v) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(v.data()), v.size()*4);
}
int main(int argc, char ** argv) {
    check(argc == 5);
    const std::string ffn = argv[2], q8 = argv[3], output = argv[4];
    check(!std::filesystem::exists(output));
    std::filesystem::create_directories(output);
    constexpr int n = 2304, m = 5120, slot = 2, padded_blocks = 80;
    const auto weight = read<char>(argv[1], ggml_row_size(GGML_TYPE_Q3_K, n)*m);
    const float route = read<float>(ffn+"/inputw1-batch1-repeat-weights.f32", 6)[slot];
    std::vector<std::vector<float>> hidden, rounded;
    for (int width : {1, 2, 4}) {
        auto h = read<float>(ffn+"/inputw1-batch"+std::to_string(width)+"-repeat-moe-hidden.f32", n*6);
        hidden.emplace_back(h.begin()+n*slot, h.begin()+n*(slot+1));
        const auto q = read<uint8_t>(q8+"/hiddenw"+std::to_string(width)+".q8_1", padded_blocks*36*6);
        std::vector<float> r(n);
        for (int j = 0; j < n; ++j) {
            const int base = (padded_blocks*slot+j/32)*36;
            const ggml_fp16_t scale = ggml_fp16_t(q[base] | (uint16_t(q[base+1])<<8));
            r[j] = ggml_fp16_to_fp32(scale)*int8_t(q[base+4+j%32]);
        }
        rounded.push_back(r);
    }
    for (int k = 1; k < 3; ++k) {
        std::vector<float> qdelta(m), fdelta(m), row(n);
        for (int i = 0; i < m; ++i) {
            ggml_get_type_traits(GGML_TYPE_Q3_K)->to_float(weight.data()+i*ggml_row_size(GGML_TYPE_Q3_K, n), row.data(), n);
            double sq = 0, sf = 0;
            for (int j = 0; j < n; ++j) {
                sq += double(row[j])*(double(rounded[k][j])-rounded[0][j]);
                sf += double(row[j])*(double(hidden[k][j])-hidden[0][j]);
            }
            qdelta[i] = float(sq*route);
            fdelta[i] = float(sf*route);
        }
        const std::string label = "w"+std::to_string(k == 1 ? 2 : 4);
        save(output+"/"+label+"-q8-delta.f32", qdelta);
        save(output+"/"+label+"-float-delta.f32", fdelta);
    }
}
