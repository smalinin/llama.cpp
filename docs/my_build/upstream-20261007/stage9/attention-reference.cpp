#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" void attention_reference(const float * q, const float * k, const float * v,
        const float * mask, const float * sinks, int nk, int nh, int d, int round_q, float * out) {
    std::vector<double> scores(nk), query(d);
    const double scale = 1.0/std::sqrt(double(d));
    for (int h = 0; h < nh; ++h) {
        const float half_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(scale)));
        for (int j = 0; j < d; ++j) {
            const float half_q = ggml_fp16_to_fp32(ggml_fp32_to_fp16(q[h*d+j]));
            query[j] = round_q ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(half_q*half_scale)) : q[h*d+j];
        }
        double maximum = sinks ? sinks[h] : -INFINITY;
        for (int i = 0; i < nk; ++i) {
            double score = 0.0;
            for (int j = 0; j < d; ++j) score += query[j]*k[i*d+j];
            scores[i] = score*(round_q ? 1.0 : scale) + mask[i];
            maximum = std::max(maximum, scores[i]);
        }
        double total = sinks ? std::exp(sinks[h]-maximum) : 0.0;
        for (double & score : scores) { score = std::exp(score-maximum); total += score; }
        for (int j = 0; j < d; ++j) {
            double value = 0.0;
            for (int i = 0; i < nk; ++i) value += scores[i]*v[i*d+j];
            out[h*d+j] = value/total;
        }
    }
}

extern "C" void bf16_projection_reference(const ggml_bf16_t * weights, const float * input,
        int n, int m, int round_input, float * out) {
    std::vector<double> values(n);
    for (int j = 0; j < n; ++j) values[j] = round_input ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(input[j])) : input[j];
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += ggml_bf16_to_fp32(weights[i*n+j])*values[j];
        out[i] = sum;
    }
}
