#include "ops.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

static float csa_e4m3(float x) {
    const float a = std::abs(x);
    if (a == 0 || !std::isfinite(a)) { return x; }
    if (a > 464) { return NAN; }
    int exponent;
    std::frexp(a, &exponent);
    const float step = std::ldexp(1.0f, std::max(exponent - 4, -9));
    return std::copysign(std::min(448.0f, std::nearbyint(a/step)*step), x);
}

static float csa_load(const ggml_tensor * t, size_t offset) {
    const char * p = (const char *) t->data + offset;
    if (t->type == GGML_TYPE_F32) { return *(const float *) p; }
    if (t->type == GGML_TYPE_F16) { return GGML_FP16_TO_FP32(*(const ggml_fp16_t *) p); }
    return GGML_BF16_TO_FP32(*(const ggml_bf16_t *) p);
}

void ggml_compute_forward_step5_csa(const ggml_compute_params * params, ggml_tensor * dst) {
    const auto * a = dst->src[0];
    const auto * b = dst->src[1];
    const int ith = params->ith, nth = params->nth;
    if (dst->op == GGML_OP_STEP5_INDEX_NORM) {
        float op[3]; std::memcpy(op, dst->op_params, sizeof(op));
        const auto * bias = dst->src[2];
        const auto * positions = (const int32_t *) dst->src[3]->data;
        for (int64_t row = ith; row < a->ne[1]*a->ne[2]; row += nth) {
            float x[256], sum[256], squares[256], normalized[256];
            for (int d = 0; d < 256; ++d) {
                x[d] = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(((const float *) a->data)[row*256+d]));
                sum[d] = x[d]; squares[d] = x[d]*x[d];
            }
            for (int stride = 128; stride; stride /= 2) {
                for (int d = 0; d < stride; ++d) { sum[d] += sum[d+stride]; squares[d] += squares[d+stride]; }
            }
            const float mean = op[2] ? sum[0]/256 : 0;
            const float rs = 1/std::sqrt(squares[0]/256-mean*mean+op[0]);
            for (int d = 0; d < 256; ++d) {
                const float weight = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(((const float *) b->data)[d])) + (op[2] ? 0 : 1);
                const float offset = bias ? GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(((const float *) bias->data)[d])) : 0;
                normalized[d] = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16((x[d]-mean)*rs*weight+offset));
            }
            auto * out = (float *) dst->data+row*256;
            std::copy(normalized, normalized+256, out);
            for (int d = 0; d < 16; ++d) {
                const float angle = positions[row/a->ne[1]]*(1.0f/std::pow(op[1], float(d)/16));
                const float c = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(std::cos(angle)));
                const float s = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(std::sin(angle)));
                out[d] = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(normalized[d]*c-normalized[d+16]*s));
                out[d+16] = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(normalized[d]*s+normalized[d+16]*c));
            }
        }
    } else
    if (dst->op == GGML_OP_STEP5_CSA_SUMMARY) {
        const auto * map = (const int32_t *) b->data;
        auto * out = (float *) dst->data;
        for (int64_t row = ith; row < dst->ne[1]*dst->ne[2]; row += nth) {
            const int64_t region = row % dst->ne[1], seq = row / dst->ne[1];
            size_t offsets[8];
            bool valid = true;
            float maximum = -INFINITY;
            for (int t = 0; t < 8; ++t) {
                const int slot = map[seq*b->ne[0] + 8*region + t];
                if (slot < 0) { valid = false; break; }
                offsets[t] = (slot/a->ne[2])*a->nb[3] + (slot%a->ne[2])*a->nb[2];
                for (int d = 0; d < 256; ++d) { maximum = std::max(maximum, csa_load(a, offsets[t] + (256+d)*a->nb[0])); }
            }
            for (int d = 0; d < 256; ++d) {
                float num = 0, den = 0;
                if (valid) {
                    for (int t = 0; t < 8; ++t) {
                        const float w = std::exp(csa_load(a, offsets[t]+(256+d)*a->nb[0])-maximum);
                        num += w*csa_load(a, offsets[t]+d*a->nb[0]); den += w;
                    }
                }
                out[row*256+d] = valid ? csa_e4m3(GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(num/den))) : NAN;
            }
        }
    } else if (dst->op == GGML_OP_STEP5_CSA_SCORES) {
        const auto * summary = dst->src[2];
        const auto * seq = (const int32_t *) dst->src[3]->data;
        const auto * pos = (const int32_t *) dst->src[4]->data;
        const auto * q = (const float *) a->data;
        const auto * w = (const float *) b->data;
        const auto * s = (const float *) summary->data;
        auto * out = (float *) dst->data;
        const int64_t nr = dst->ne[0];
        for (int64_t row = ith; row < ggml_nelements(dst); row += nth) {
            const int64_t r = row % nr, group = row/nr % 4, qi = row/(nr*4);
            float score = -INFINITY;
            if (r < pos[qi]/8 && std::isfinite(s[(seq[qi]*nr+r)*256])) {
                score = 0;
                for (int h = 0; h < 4; ++h) {
                    const int64_t head = qi*16+group*4+h;
                    float dot = 0;
                    for (int d = 0; d < 256; ++d) { dot += csa_e4m3(q[head*256+d])*s[(seq[qi]*nr+r)*256+d]; }
                    score += std::max(0.0f, dot)*w[head]*0.5f;
                }
            }
            out[row] = score;
        }
    } else if (dst->op == GGML_OP_STEP5_CSA_TOPK) {
        const auto * pos = (const int32_t *) b->data;
        const int topk = ggml_get_op_params_i32(dst, 0), width = dst->ne[0];
        for (int64_t row = ith; row < dst->ne[1]*dst->ne[2]; row += nth) {
            const int qi = row/4, count = std::min<int64_t>(pos[qi]/8, a->ne[0]);
            const float * scores = (const float *) a->data + row*a->ne[0];
            auto * out = (int32_t *) dst->data + row*width;
            std::fill(out, out+width, -1);
            std::vector<int> ids(count);
            std::iota(ids.begin(), ids.end(), 0);
            const int take = std::min(topk, count);
            std::partial_sort(ids.begin(), ids.begin()+take, ids.end(), [&](int x, int y) {
                return scores[x] > scores[y] || (scores[x] == scores[y] && x < y);
            });
            std::sort(ids.begin(), ids.begin()+take);
            int n = 0;
            for (int j = 0; j < take; ++j) { for (int t = 0; t < 8; ++t) { out[n++] = ids[j]*8+t; } }
            for (int t = (pos[qi]/8)*8; t <= pos[qi]; ++t) { out[n++] = t; }
        }
    } else {
        const auto * v = dst->src[2];
        const auto * selected = dst->src[3];
        const auto * slots = dst->src[4];
        const auto * seq = (const int32_t *) dst->src[5]->data;
        const auto * map = (const int32_t *) slots->data;
        float scale; std::memcpy(&scale, dst->op_params, sizeof(scale));
        for (int64_t row = ith; row < a->ne[1]*a->ne[2]; row += nth) {
            const int qi = row/a->ne[1], head = row%a->ne[1], group = head/(a->ne[1]/4), dim = a->ne[0];
            const auto * ids = (const int32_t *) selected->data + (qi*4+group)*selected->ne[0];
            const auto * q = (const float *) ((const char *) a->data + qi*a->nb[2]+head*a->nb[1]);
            float acc[256] = {}, maximum = -INFINITY, total = 0;
            for (int i = 0; i < selected->ne[0]; ++i) {
                const int p = ids[i];
                if (p < 0 || p >= slots->ne[0]) { continue; }
                const int slot = map[seq[qi]*slots->ne[0]+p];
                if (slot < 0) { continue; }
                const size_t ko = (slot/b->ne[2])*b->nb[3]+(slot%b->ne[2])*b->nb[2]+group*b->nb[1];
                const size_t vo = (slot/v->ne[2])*v->nb[3]+(slot%v->ne[2])*v->nb[2]+group*v->nb[1];
                float dot = 0;
                for (int d = 0; d < dim; ++d) { dot += q[d]*csa_load(b, ko+d*b->nb[0]); }
                dot *= scale;
                const float m = std::max(maximum, dot), old = std::exp(maximum-m), weight = std::exp(dot-m);
                total = total*old+weight; maximum = m;
                for (int d = 0; d < dim; ++d) { acc[d] = acc[d]*old+weight*csa_load(v, vo+d*v->nb[0]); }
            }
            auto * out = (float *) dst->data + row*dim;
            for (int d = 0; d < dim; ++d) { out[d] = total > 0 ? acc[d]/total : 0; }
        }
    }
}
