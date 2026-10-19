#pragma once

#include "ggml.h"

#include <cmath>

static inline ggml_tensor * dsv41_engram_dot(
        ggml_context * ctx, ggml_tensor * h, ggml_tensor * key,
        ggml_tensor * weight, ggml_tensor * one, float eps) {
    GGML_ASSERT(h->type == GGML_TYPE_F32 && key->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(h, key));
    auto inverse_rms = [&](ggml_tensor * x) {
        ggml_tensor * variance = ggml_scale_bias(ctx, ggml_mean(ctx, ggml_sqr(ctx, x)), 1.0f, eps);
        return ggml_div(ctx, ggml_repeat(ctx, one, variance), ggml_sqrt(ctx, variance));
    };
    ggml_tensor * rstd = ggml_mul(ctx, inverse_rms(h), inverse_rms(key));
    ggml_tensor * dot = ggml_sum_rows(ctx, ggml_mul(ctx, ggml_mul(ctx, h, weight), key));
    return ggml_scale(ctx, ggml_mul(ctx, dot, rstd), 1.0f/sqrtf((float) h->ne[0]));
}
