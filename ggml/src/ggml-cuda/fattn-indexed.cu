#include "common.cuh"
#include "fattn-indexed.cuh"

template<bool Q8_0>
static __device__ __forceinline__ float load_indexed_kv(const char * row, int d) {
    if constexpr (Q8_0) {
        const block_q8_0 * blocks = (const block_q8_0 *) row;
        return __half2float(blocks[d/QK8_0].d)*blocks[d/QK8_0].qs[d%QK8_0];
    } else {
        return __half2float(((const half *) row)[d]);
    }
}

template<int D, int N_WARPS, bool Q8_0>
static __global__ void flash_attn_ext_indexed(
        const float * __restrict__ q,
        const void  * __restrict__ k,
        const void  * __restrict__ v,
        const half  * __restrict__ mask,
        const int   * __restrict__ indices,
              float * __restrict__ dst,
        float scale,
        int n_selected,
        int n_query,
        int n_head,
        int n_kv,
        int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
        int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
        int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
        int64_t d_nb1, int64_t d_nb2, int64_t d_nb3) {
    const int iq   = blockIdx.x;
    const int head = blockIdx.y;
    const int seq  = blockIdx.z;
    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;

    __shared__ float partial_max[N_WARPS];
    __shared__ float partial_sum[N_WARPS];
    __shared__ float partial_out[N_WARPS*D];

    const float * q_row = (const float *) ((const char *) q +
            iq*q_nb1 + head*q_nb2 + seq*q_nb3);
    float * dst_row = (float *) ((char *) dst +
            head*d_nb1 + iq*d_nb2 + seq*d_nb3);

    constexpr int n_per_lane = D/WARP_SIZE;
    float qv[n_per_lane];
    float out[n_per_lane] = { 0.0f };

#pragma unroll
    for (int j = 0; j < n_per_lane; ++j) {
        qv[j] = q_row[lane + j*WARP_SIZE];
    }

    float max_score = -INFINITY;
    float sum = 0.0f;
    const int query_base = (seq*n_query + iq)*n_selected;

    for (int is = warp; is < n_selected; is += N_WARPS) {
        const float mv = __half2float(mask[query_base + is]);
        if (mv == -INFINITY) {
            continue;
        }

        const int cell = indices[query_base + is];
        if (cell < 0 || cell >= n_kv) {
            continue;
        }
        const char * k_row = (const char *) k + cell*k_nb1 + seq*k_nb3;

        float score = 0.0f;
#pragma unroll
        for (int j = 0; j < n_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            score += qv[j]*load_indexed_kv<Q8_0>(k_row, d);
        }
        score = warp_reduce_sum(score)*scale + mv;

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (lane == 0) {
            if (score > max_score) {
                old_scale = expf(max_score - score);
                max_score = score;
            } else {
                value_scale = expf(score - max_score);
            }
            sum = sum*old_scale + value_scale;
        }
        old_scale   = __shfl_sync(0xffffffff, old_scale,   0, WARP_SIZE);
        value_scale = __shfl_sync(0xffffffff, value_scale, 0, WARP_SIZE);

        const char * v_row = (const char *) v + cell*v_nb1 + seq*v_nb3;
#pragma unroll
        for (int j = 0; j < n_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            out[j] = out[j]*old_scale + load_indexed_kv<Q8_0>(v_row, d)*value_scale;
        }
    }

#pragma unroll
    for (int j = 0; j < n_per_lane; ++j) {
        partial_out[warp*D + lane + j*WARP_SIZE] = out[j];
    }
    if (lane == 0) {
        partial_max[warp] = max_score;
        partial_sum[warp] = sum;
    }
    __syncthreads();

    if (warp != 0) {
        return;
    }

    float total_max = -INFINITY;
    float total_sum = 0.0f;
    if (lane == 0) {
#pragma unroll
        for (int iw = 0; iw < N_WARPS; ++iw) {
            total_max = fmaxf(total_max, partial_max[iw]);
        }
#pragma unroll
        for (int iw = 0; iw < N_WARPS; ++iw) {
            if (partial_sum[iw] != 0.0f) {
                total_sum += partial_sum[iw]*expf(partial_max[iw] - total_max);
            }
        }
    }
    total_max = __shfl_sync(0xffffffff, total_max, 0, WARP_SIZE);
    total_sum = __shfl_sync(0xffffffff, total_sum, 0, WARP_SIZE);

#pragma unroll
    for (int j = 0; j < n_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        float value = 0.0f;
#pragma unroll
        for (int iw = 0; iw < N_WARPS; ++iw) {
            if (partial_sum[iw] != 0.0f) {
                value += partial_out[iw*D + d]*expf(partial_max[iw] - total_max);
            }
        }
        dst_row[d] = total_sum == 0.0f ? 0.0f : value/total_sum;
    }
}

template<int N_WARPS, bool Q8_0>
static void launch_flash_attn_ext_indexed(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * mask,
        const ggml_tensor * indices,
        ggml_tensor * dst,
        float scale) {
    const dim3 blocks(q->ne[1], q->ne[2], q->ne[3]);
    const dim3 threads(N_WARPS*WARP_SIZE, 1, 1);
    flash_attn_ext_indexed<512, N_WARPS, Q8_0><<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) q->data,
            k->data,
            v->data,
            (const half  *) mask->data,
            (const int   *) indices->data,
            (float       *) dst->data,
            scale, indices->ne[0], q->ne[1], q->ne[2], k->ne[1],
            q->nb[1], q->nb[2], q->nb[3],
            k->nb[1], k->nb[2], k->nb[3],
            v->nb[1], v->nb[2], v->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_flash_attn_ext_indexed_supported(const ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sinks   = dst->src[4];
    const ggml_tensor * indices = dst->src[5];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const bool kv_type = (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) ||
                         (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0);

    return indices != nullptr && dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && kv_type &&
        mask != nullptr && mask->type == GGML_TYPE_F16 &&
        indices->type == GGML_TYPE_I32 && sinks == nullptr &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(indices) &&
        q->ne[0] == 512 && k->ne[0] == 512 && v->ne[0] == 512 &&
        k->ne[2] == 1 && v->ne[2] == 1 &&
        mask->ne[0] == indices->ne[0] && mask->ne[1] == q->ne[1] &&
        mask->ne[2] == 1 && mask->ne[3] == q->ne[3] &&
        indices->ne[1] == q->ne[1] && indices->ne[2] == q->ne[3] &&
        k->ne[3] == q->ne[3] && v->ne[3] == q->ne[3] &&
        max_bias == 0.0f && logit_softcap == 0.0f;
}

void ggml_cuda_flash_attn_ext_indexed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_indexed_supported(dst));

    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * indices = dst->src[5];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));

    if (k->type == GGML_TYPE_Q8_0) {
        launch_flash_attn_ext_indexed<8, true>(ctx, q, k, v, mask, indices, dst, scale);
    } else {
        launch_flash_attn_ext_indexed<8, false>(ctx, q, k, v, mask, indices, dst, scale);
    }
}
