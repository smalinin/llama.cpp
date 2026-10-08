#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * text) { if (!ok) throw std::runtime_error(text); }

template<typename T> static std::vector<T> read(const std::string & path, size_t count) {
    std::vector<T> result(count);
    std::ifstream input(path, std::ios::binary);
    check(bool(input.read(reinterpret_cast<char *>(result.data()), count*sizeof(T))), "read failed");
    return result;
}

int main(int argc, char ** argv) {
    check(argc == 6, "usage: projection-q8-reference INPUT TYPE N M OUTPUT");
    const std::string root = argv[1], output = argv[5];
    const auto type = static_cast<ggml_type>(std::stoi(argv[2]));
    const int n = std::stoi(argv[3]), m = std::stoi(argv[4]);
    check(n % 32 == 0, "input length must be divisible by 32");
    check(!std::filesystem::exists(output), "output exists");
    std::filesystem::create_directories(output);
    const auto weights = read<char>(root+"/weight.bin", ggml_row_size(type, n)*m);
    std::vector<float> wf(size_t(n)*m);
    for (int row = 0; row < m; ++row) ggml_get_type_traits(type)->to_float(weights.data()+row*ggml_row_size(type, n), wf.data()+size_t(row)*n, n);
    const bool normalize = std::filesystem::exists(root+"/norm.f32");
    const auto norm = normalize ? read<float>(root+"/norm.f32", m) : std::vector<float>();
    float epsilon = 0;
    if (normalize) { std::ifstream f(root+"/epsilon.txt"); check(bool(f >> epsilon), "missing epsilon"); }
    std::ofstream report(output+"/results.jsonl");
    for (int width : {1, 2, 4}) {
        const auto x = read<float>(root+"/input-w"+std::to_string(width)+".f32", n);
        const auto captured = read<float>(root+"/captured-w"+std::to_string(width)+".f32", m);
        std::vector<float> rounded(n), y(m);
        for (int start = 0; start < n; start += 32) {
            float amax = 0;
            for (int j = start; j < start+32; ++j) amax = std::max(amax, std::abs(x[j]));
            const float d = amax/127.0f;
            const float half_d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            for (int j = start; j < start+32; ++j) rounded[j] = d == 0 ? 0 : std::round(x[j]/d)*half_d;
        }
        for (int row = 0; row < m; ++row) {
            double sum = 0;
            for (int j = 0; j < n; ++j) sum += double(wf[size_t(row)*n+j])*rounded[j];
            y[row] = float(sum);
        }
        if (normalize) {
            double squares = 0;
            for (float v : y) squares += double(v)*v;
            const double scale = 1/std::sqrt(squares/m+epsilon);
            for (int row = 0; row < m; ++row) y[row] *= scale*norm[row];
        }
        double maximum = 0, squares = 0;
        for (int row = 0; row < m; ++row) {
            const double d = double(y[row])-captured[row];
            maximum = std::max(maximum, std::abs(d));
            squares += d*d;
        }
        const std::string path = output+"/inputw"+std::to_string(width)+".f32";
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(y.data()), y.size()*4);
        report << "{\"input_width\":" << width << ",\"max_vs_capture\":" << maximum << ",\"rms_vs_capture\":" << std::sqrt(squares/m) << "}\n";
    }
}
