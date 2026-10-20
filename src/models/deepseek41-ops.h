#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>

static inline ggml_tensor * dsv41_order_indices(ggml_context * ctx, ggml_tensor * indices) {
    GGML_ASSERT(ggml_is_vector(indices));
    ggml_tensor * order = ggml_argsort(ctx, ggml_cast(ctx, indices, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
    return ggml_reshape_1d(ctx,
            ggml_get_rows(ctx, ggml_reshape_2d(ctx, indices, 1, indices->ne[0]), order), indices->ne[0]);
}

static inline void dsv41_select_kv(ggml_context * ctx, ggml_tensor * & k, ggml_tensor * & mask,
        ggml_tensor * indices, int64_t n_raw, int64_t n_swa) {
    GGML_ASSERT(k->type == GGML_TYPE_F16 && k->ne[1] == 1 && k->ne[3] == 1);
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_vector(mask) && mask->ne[0] == k->ne[2]);
    GGML_ASSERT(ggml_is_vector(indices));
    const int64_t n = k->ne[2];
    const int64_t selected = indices->ne[0];
    ggml_tensor * rows = ggml_view_2d(ctx, k, k->ne[0], n, k->nb[2], 0);
    k = ggml_get_rows(ctx, rows, indices);
    k = ggml_cast(ctx, ggml_reshape_4d(ctx, k, k->ne[0], 1, selected, 1), GGML_TYPE_F16);
    mask = ggml_get_rows(ctx, ggml_reshape_2d(ctx, mask, 1, n), indices);
    mask = ggml_cast(ctx, ggml_reshape_4d(ctx, mask, selected, 1, 1, 1), GGML_TYPE_F16);

    // Retain sparse MMA accumulation; dense MMA rounding changes quantized downstream activations.
    const int64_t padding = std::max<int64_t>(0, std::max<int64_t>(4096, 2*(n_swa + selected)) - n_raw - selected);
    if (padding > 0) {
        k = ggml_concat(ctx, k,
                ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k->ne[0], 1, padding, 1), 0.0f), 2);
        mask = ggml_concat(ctx, mask,
                ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, padding, 1, 1, 1), -INFINITY), 0);
    }
}

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
