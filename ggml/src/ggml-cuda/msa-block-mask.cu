#include "msa-block-mask.cuh"
#include "fattn-common.cuh"
#include "top-k.cuh"

#include <cfloat>
#include <climits>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

#include <mma.h>
namespace wmma = nvcuda::wmma;

template <ggml_type TYPE_K>
static __global__ void msa_block_scores_wmma_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ pos_cell,
        const char * __restrict__ q_pos,
        const char * __restrict__ mask,
        float * __restrict__ scores,
        const int n_heads,
        const int n_tokens,
        const int n_streams,
        const int n_kv,
        const int n_pos,
        const int n_blocks,
        const int n_local_blocks,
        const size_t q_nb1,
        const size_t q_nb2,
        const size_t q_nb3,
        const size_t k_nb2,
        const size_t k_nb3,
        const size_t pos_nb1,
        const size_t q_pos_nb1,
        const size_t mask_nb1,
        const size_t mask_nb3) {
    constexpr int D            = 128;
    constexpr int BLOCK_SIZE   = 128;
    constexpr int Q_TILE       = 16;
    constexpr int N_WARPS      = 8;
    constexpr int THREADS      = N_WARPS * WARP_SIZE;

    const int block  = blockIdx.x;
    const int q0     = blockIdx.y * Q_TILE;
    const int stream = blockIdx.z;
    const int tid    = threadIdx.x + WARP_SIZE * threadIdx.y;
    const int warp   = threadIdx.y;

    __shared__ half  q_shared[Q_TILE][D];
    __shared__ half  k_shared[BLOCK_SIZE][D];
    __shared__ int   k_valid[BLOCK_SIZE];
    __shared__ float qk_shared[N_WARPS][Q_TILE][16];

    const int n_q_rows = n_heads * n_tokens;

    for (int i = tid; i < Q_TILE * D; i += THREADS) {
        const int qr  = i / D;
        const int dim = i % D;
        const int row = q0 + qr;
        if (row < n_q_rows) {
            const int head  = row % n_heads;
            const int token = row / n_heads;
            const float value = *(const float *) (q + dim * sizeof(float) +
                    head*q_nb1 + token*q_nb2 + stream*q_nb3);
            q_shared[qr][dim] = __float2half(value);
        } else {
            q_shared[qr][dim] = __float2half(0.0f);
        }
    }

    if (tid < BLOCK_SIZE) {
        const int pos = block * BLOCK_SIZE + tid;
        const int cell = pos < n_pos
                ? *(const int32_t *) (pos_cell + pos * sizeof(int32_t) + stream*pos_nb1)
                : -1;
        k_valid[tid] = cell >= 0 && cell < n_kv ? cell : -1;
    }

    for (int i = tid; i < BLOCK_SIZE * (D / 4); i += THREADS) {
        const int kt   = i / (D / 4);
        const int dim4 = (i % (D / 4)) * 4;
        const int pos  = block * BLOCK_SIZE + kt;
        const int cell = pos < n_pos
                ? *(const int32_t *) (pos_cell + pos * sizeof(int32_t) + stream*pos_nb1)
                : -1;

        if (cell >= 0 && cell < n_kv) {
            const void * k_row = k + cell*k_nb2 + stream*k_nb3;
            if constexpr (TYPE_K == GGML_TYPE_F32) {
                const float4 value = *(const float4 *) ((const char *) k_row + dim4*sizeof(float));
                ((half2 *) &k_shared[kt][dim4])[0] = __float22half2_rn(make_float2(value.x, value.y));
                ((half2 *) &k_shared[kt][dim4])[1] = __float22half2_rn(make_float2(value.z, value.w));
            } else if constexpr (TYPE_K == GGML_TYPE_BF16) {
                float4 value;
                constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                dequantize_k(k_row, &value, dim4);
                ((half2 *) &k_shared[kt][dim4])[0] = __float22half2_rn(make_float2(value.x, value.y));
                ((half2 *) &k_shared[kt][dim4])[1] = __float22half2_rn(make_float2(value.z, value.w));
            } else {
                constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
                dequantize_k(k_row, &k_shared[kt][dim4], dim4);
            }
        } else {
            *(int2 *) &k_shared[kt][dim4] = make_int2(0, 0);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

#pragma unroll
    for (int dim = 0; dim < D; dim += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
        wmma::load_matrix_sync(q_frag, &q_shared[0][dim], D);
        wmma::load_matrix_sync(k_frag, &k_shared[warp * 16][dim], D);
        wmma::mma_sync(acc, q_frag, k_frag, acc);
    }

    wmma::store_matrix_sync(&qk_shared[warp][0][0], acc, 16, wmma::mem_row_major);
    __syncthreads();

    if (tid < Q_TILE && q0 + tid < n_q_rows) {
        const int row   = q0 + tid;
        const int token = row / n_heads;
        const int32_t query_pos = *(const int32_t *) (q_pos + token*sizeof(int32_t) + stream*q_pos_nb1);

        float score = -INFINITY;
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int kt   = w * 16 + j;
                const int cell = k_valid[kt];
                const int pos  = block * BLOCK_SIZE + kt;
                const float mask_value = cell >= 0
                        ? __half2float(*(const half *) (mask + cell*sizeof(half) +
                                token*mask_nb1 + stream*mask_nb3))
                        : -INFINITY;
                if (isfinite(mask_value) && pos <= query_pos) {
                    score = fmaxf(score, qk_shared[w][tid][j]);
                }
            }
        }

        const int query_block = query_pos / BLOCK_SIZE;
        const int local_first = query_block - n_local_blocks + 1;
        if (isfinite(score) && block >= local_first && block <= query_block) {
            score = FLT_MAX;
        }

        scores[((int64_t) stream*n_q_rows + row)*n_blocks + block] = score;
    }
}

template <ggml_type TYPE_K>
static __global__ void msa_block_top_k_wmma_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ pos_cell,
        const char * __restrict__ q_pos,
        const char * __restrict__ mask,
        char * __restrict__ dst,
        const int n_heads,
        const int n_tokens,
        const int n_streams,
        const int n_kv,
        const int n_pos,
        const int n_blocks,
        const int top_k,
        const int n_local_blocks,
        const size_t q_nb1,
        const size_t q_nb2,
        const size_t q_nb3,
        const size_t k_nb2,
        const size_t k_nb3,
        const size_t pos_nb1,
        const size_t q_pos_nb1,
        const size_t mask_nb1,
        const size_t mask_nb3,
        const size_t dst_nb1,
        const size_t dst_nb2,
        const size_t dst_nb3) {
    constexpr int D            = 128;
    constexpr int BLOCK_SIZE   = 128;
    constexpr int Q_TILE       = 8;
    constexpr int WMMA_Q_TILE  = 16;
    constexpr int N_WARPS      = 8;
    constexpr int MAX_TOP_K    = 32;
    constexpr int THREADS      = N_WARPS * WARP_SIZE;

    const int q0     = blockIdx.x * Q_TILE;
    const int stream = blockIdx.z;
    const int tid    = threadIdx.x + WARP_SIZE * threadIdx.y;
    const int warp   = threadIdx.y;
    const int n_q_rows = n_heads * n_tokens;

    __shared__ half  q_shared[WMMA_Q_TILE][D];
    __shared__ half  k_shared[BLOCK_SIZE][D];
    __shared__ int   k_cell[BLOCK_SIZE];
    __shared__ float qk_shared[N_WARPS][WMMA_Q_TILE][16];
    __shared__ float top_scores[Q_TILE][MAX_TOP_K];
    __shared__ int   top_indices[Q_TILE][MAX_TOP_K];

    for (int i = tid; i < WMMA_Q_TILE * D; i += THREADS) {
        const int qr  = i / D;
        const int dim = i % D;
        const int row = q0 + qr;
        if (qr < Q_TILE && row < n_q_rows) {
            const int head  = row % n_heads;
            const int token = row / n_heads;
            const float value = *(const float *) (q + dim * sizeof(float) +
                    head*q_nb1 + token*q_nb2 + stream*q_nb3);
            q_shared[qr][dim] = __float2half(value);
        } else {
            q_shared[qr][dim] = __float2half(0.0f);
        }
    }
    for (int i = tid; i < Q_TILE*MAX_TOP_K; i += THREADS) {
        top_scores[i/MAX_TOP_K][i%MAX_TOP_K] = -INFINITY;
        top_indices[i/MAX_TOP_K][i%MAX_TOP_K] = INT_MAX;
    }
    __syncthreads();

    for (int block = 0; block < n_blocks; ++block) {
        if (tid < BLOCK_SIZE) {
            const int pos = block * BLOCK_SIZE + tid;
            const int cell = pos < n_pos
                    ? *(const int32_t *) (pos_cell + pos * sizeof(int32_t) + stream*pos_nb1)
                    : -1;
            k_cell[tid] = cell >= 0 && cell < n_kv ? cell : -1;
        }

        for (int i = tid; i < BLOCK_SIZE * (D / 4); i += THREADS) {
            const int kt   = i / (D / 4);
            const int dim4 = (i % (D / 4)) * 4;
            const int pos  = block * BLOCK_SIZE + kt;
            const int cell = pos < n_pos
                    ? *(const int32_t *) (pos_cell + pos * sizeof(int32_t) + stream*pos_nb1)
                    : -1;

            if (cell >= 0 && cell < n_kv) {
                const void * k_row = k + cell*k_nb2 + stream*k_nb3;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    const float4 value = *(const float4 *) ((const char *) k_row + dim4*sizeof(float));
                    ((half2 *) &k_shared[kt][dim4])[0] = __float22half2_rn(make_float2(value.x, value.y));
                    ((half2 *) &k_shared[kt][dim4])[1] = __float22half2_rn(make_float2(value.z, value.w));
                } else if constexpr (TYPE_K == GGML_TYPE_BF16) {
                    float4 value;
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &value, dim4);
                    ((half2 *) &k_shared[kt][dim4])[0] = __float22half2_rn(make_float2(value.x, value.y));
                    ((half2 *) &k_shared[kt][dim4])[1] = __float22half2_rn(make_float2(value.z, value.w));
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
                    dequantize_k(k_row, &k_shared[kt][dim4], dim4);
                }
            } else {
                *(int2 *) &k_shared[kt][dim4] = make_int2(0, 0);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

#pragma unroll
        for (int dim = 0; dim < D; dim += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(q_frag, &q_shared[0][dim], D);
            wmma::load_matrix_sync(k_frag, &k_shared[warp * 16][dim], D);
            wmma::mma_sync(acc, q_frag, k_frag, acc);
        }

        wmma::store_matrix_sync(&qk_shared[warp][0][0], acc, WMMA_Q_TILE, wmma::mem_row_major);
        __syncthreads();

        if (tid < Q_TILE && q0 + tid < n_q_rows) {
            const int row   = q0 + tid;
            const int token = row / n_heads;
            const int32_t query_pos = *(const int32_t *) (q_pos + token*sizeof(int32_t) + stream*q_pos_nb1);

            float score = -INFINITY;
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int kt   = w * 16 + j;
                    const int cell = k_cell[kt];
                    const int pos  = block * BLOCK_SIZE + kt;
                    const float mask_value = cell >= 0
                            ? __half2float(*(const half *) (mask + cell*sizeof(half) +
                                    token*mask_nb1 + stream*mask_nb3))
                            : -INFINITY;
                    if (isfinite(mask_value) && pos <= query_pos) {
                        score = fmaxf(score, qk_shared[w][tid][j]);
                    }
                }
            }

            const int query_block = query_pos / BLOCK_SIZE;
            const int local_first = query_block - n_local_blocks + 1;
            if (isfinite(score) && block >= local_first && block <= query_block) {
                score = FLT_MAX;
            }

            int insert = top_k;
            while (insert > 0) {
                const float prev_score = top_scores[tid][insert - 1];
                const int prev_index = top_indices[tid][insert - 1];
                if (score < prev_score || (score == prev_score && block >= prev_index)) {
                    break;
                }
                --insert;
            }
            if (insert < top_k) {
                for (int rank = top_k - 1; rank > insert; --rank) {
                    top_scores[tid][rank] = top_scores[tid][rank - 1];
                    top_indices[tid][rank] = top_indices[tid][rank - 1];
                }
                top_scores[tid][insert] = score;
                top_indices[tid][insert] = block;
            }
        }
        __syncthreads();
    }

    if (tid < Q_TILE && q0 + tid < n_q_rows) {
        const int row   = q0 + tid;
        const int head  = row % n_heads;
        const int token = row / n_heads;
        int32_t * dst_row = (int32_t *) (dst + head*dst_nb1 + token*dst_nb2 + stream*dst_nb3);
        for (int rank = 0; rank < top_k; ++rank) {
            dst_row[rank] = top_indices[tid][rank];
        }
    }
}

#else

template <ggml_type TYPE_K>
static __global__ void msa_block_scores_wmma_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ pos_cell,
        const char * __restrict__ q_pos,
        const char * __restrict__ mask,
        float * __restrict__ scores,
        const int n_heads, const int n_tokens, const int n_streams, const int n_kv, const int n_pos,
        const int n_blocks, const int n_local_blocks, const size_t q_nb1, const size_t q_nb2,
        const size_t q_nb3, const size_t k_nb2, const size_t k_nb3, const size_t pos_nb1,
        const size_t q_pos_nb1, const size_t mask_nb1, const size_t mask_nb3) {
    GGML_UNUSED_VARS(q, k, pos_cell, q_pos, mask, scores, n_heads, n_tokens, n_streams, n_kv, n_pos,
            n_blocks, n_local_blocks, q_nb1, q_nb2, q_nb3, k_nb2, k_nb3, pos_nb1, q_pos_nb1,
            mask_nb1, mask_nb3);
    NO_DEVICE_CODE;
}

template <ggml_type TYPE_K>
static __global__ void msa_block_top_k_wmma_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ pos_cell,
        const char * __restrict__ q_pos, const char * __restrict__ mask, char * __restrict__ dst,
        const int n_heads, const int n_tokens, const int n_streams, const int n_kv, const int n_pos,
        const int n_blocks, const int top_k, const int n_local_blocks, const size_t q_nb1,
        const size_t q_nb2, const size_t q_nb3, const size_t k_nb2, const size_t k_nb3,
        const size_t pos_nb1, const size_t q_pos_nb1, const size_t mask_nb1, const size_t mask_nb3,
        const size_t dst_nb1, const size_t dst_nb2, const size_t dst_nb3) {
    GGML_UNUSED_VARS(q, k, pos_cell, q_pos, mask, dst, n_heads, n_tokens, n_streams, n_kv, n_pos,
            n_blocks, top_k, n_local_blocks, q_nb1, q_nb2, q_nb3, k_nb2, k_nb3, pos_nb1, q_pos_nb1,
            mask_nb1, mask_nb3, dst_nb1, dst_nb2, dst_nb3);
    NO_DEVICE_CODE;
}

#endif

#define MSA_BLOCK_TOP_K_CASE(type_k) \
    case type_k: \
        msa_block_scores_wmma_kernel<type_k><<<grid, threads, 0, ctx.stream()>>>( \
                (const char *) q->data, (const char *) k->data, \
                (const char *) pos_cell->data, (const char *) q_pos->data, (const char *) mask->data, scores, \
                n_heads, n_tokens, n_streams, n_kv, n_pos, n_blocks, n_local_blocks, \
                q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3], pos_cell->nb[1], q_pos->nb[1], \
                mask->nb[1], mask->nb[3]); \
        break;

#define MSA_BLOCK_TOP_K_FUSED_CASE(type_k) \
    case type_k: \
        msa_block_top_k_wmma_kernel<type_k><<<grid, threads, 0, ctx.stream()>>>( \
                (const char *) q->data, (const char *) k->data, \
                (const char *) pos_cell->data, (const char *) q_pos->data, \
                (const char *) mask->data, (char *) dst->data, \
                n_heads, n_tokens, n_streams, n_kv, n_pos, n_blocks, dst->ne[0], n_local_blocks, \
                q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3], pos_cell->nb[1], q_pos->nb[1], \
                mask->nb[1], mask->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]); \
        break;

#endif

void ggml_cuda_op_msa_block_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q        = dst->src[0];
    const ggml_tensor * k        = dst->src[1];
    const ggml_tensor * pos_cell = dst->src[2];
    const ggml_tensor * q_pos    = dst->src[3];
    const ggml_tensor * mask     = dst->src[4];

    const int n_heads   = q->ne[1];
    const int n_tokens  = q->ne[2];
    const int n_streams = q->ne[3];
    const int n_kv      = k->ne[2];
    const int n_pos     = pos_cell->ne[0];
    const int block_size = ggml_get_op_params_i32(dst, 0);
    const int n_local_blocks = ggml_get_op_params_i32(dst, 1);
    const int n_blocks = n_pos / block_size;
    const int n_rows = n_heads * n_tokens * n_streams;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (n_heads*n_tokens >= 512 && dst->ne[0] <= 32) {
        constexpr int Q_TILE = 8;
        constexpr int N_WARPS = 8;
        const dim3 threads(WARP_SIZE, N_WARPS);
        const dim3 grid((n_heads*n_tokens + Q_TILE - 1) / Q_TILE, 1, n_streams);

        switch (k->type) {
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_F32)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_BF16)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_F16)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_Q8_0)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_Q5_1)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_Q5_0)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_Q4_1)
            MSA_BLOCK_TOP_K_FUSED_CASE(GGML_TYPE_Q4_0)
            default:
                GGML_ABORT("unsupported MSA index K type");
        }
        return;
    }
#endif

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> scores_alloc(pool, (size_t) n_blocks * n_rows);
    float * scores = scores_alloc.get();

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    constexpr int Q_TILE = 16;
    constexpr int N_WARPS = 8;
    const dim3 threads(WARP_SIZE, N_WARPS);
    const dim3 grid(n_blocks, (n_heads*n_tokens + Q_TILE - 1) / Q_TILE, n_streams);

    switch (k->type) {
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_F32)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_BF16)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_F16)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_Q8_0)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_Q5_1)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_Q5_0)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_Q4_1)
        MSA_BLOCK_TOP_K_CASE(GGML_TYPE_Q4_0)
        default:
            GGML_ABORT("unsupported MSA index K type");
    }
#else
    GGML_UNUSED_VARS(q, k, pos_cell, q_pos, mask, scores, n_kv, n_pos, n_local_blocks);
    GGML_ABORT("MSA block Top-K requires NVIDIA tensor cores");
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#undef MSA_BLOCK_TOP_K_CASE
#undef MSA_BLOCK_TOP_K_FUSED_CASE
#endif

    ggml_tensor score_tensor = {};
    score_tensor.type = GGML_TYPE_F32;
    score_tensor.ne[0] = n_blocks;
    score_tensor.ne[1] = n_heads;
    score_tensor.ne[2] = n_tokens;
    score_tensor.ne[3] = n_streams;
    score_tensor.nb[0] = sizeof(float);
    score_tensor.nb[1] = score_tensor.nb[0] * score_tensor.ne[0];
    score_tensor.nb[2] = score_tensor.nb[1] * score_tensor.ne[1];
    score_tensor.nb[3] = score_tensor.nb[2] * score_tensor.ne[2];
    score_tensor.data  = scores;

    ggml_tensor top_k = *dst;
    top_k.src[0] = &score_tensor;
    ggml_cuda_op_top_k(ctx, &top_k);
}

bool ggml_cuda_msa_block_top_k_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * q        = dst->src[0];
    const ggml_tensor * k        = dst->src[1];
    const ggml_tensor * pos_cell = dst->src[2];
    const ggml_tensor * q_pos    = dst->src[3];
    const ggml_tensor * mask     = dst->src[4];

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32 || q->ne[0] != 128 ||
            k->ne[0] != 128 || k->ne[1] != 1 || q->ne[2] != q_pos->ne[0] ||
            q->ne[3] != k->ne[3] || q->ne[3] != pos_cell->ne[1] || q->ne[3] != q_pos->ne[1]) {
        return false;
    }
    if (pos_cell->type != GGML_TYPE_I32 || q_pos->type != GGML_TYPE_I32 || mask->type != GGML_TYPE_F16 ||
            pos_cell->ne[2] != 1 || pos_cell->ne[3] != 1 || q_pos->ne[2] != 1 || q_pos->ne[3] != 1 ||
            mask->ne[0] != k->ne[2] || mask->ne[1] != q->ne[2] ||
            mask->ne[2] != 1 || mask->ne[3] != q->ne[3] ||
            dst->ne[1] != q->ne[1] || dst->ne[2] != q->ne[2] || dst->ne[3] != q->ne[3]) {
        return false;
    }
    if (q->ne[1] > INT_MAX || q->ne[2] > INT_MAX || q->ne[3] > INT_MAX ||
            pos_cell->ne[0] > INT_MAX || k->ne[2] > INT_MAX ||
            q->ne[1]*q->ne[2] > INT_MAX/q->ne[3]) {
        return false;
    }
    const int block_size = ggml_get_op_params_i32(dst, 0);
    const int n_local_blocks = ggml_get_op_params_i32(dst, 1);
    if (block_size != 128 || pos_cell->ne[0] % block_size != 0 || dst->ne[0] <= 0 ||
            dst->ne[0] > pos_cell->ne[0]/block_size || n_local_blocks < 0 || n_local_blocks > dst->ne[0]) {
        return false;
    }
    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) ||
            !ggml_is_contiguous(pos_cell) || !ggml_is_contiguous(q_pos) ||
            mask->nb[0] != sizeof(half) || !ggml_is_contiguous_rows(mask) || !ggml_is_contiguous(dst)) {
        return false;
    }

    switch (k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ void msa_dequantize_4(
        const ggml_type type, const void * row, half * dst, const int dim) {
    switch (type) {
        case GGML_TYPE_F32: {
            const float4 value = *(const float4 *) ((const char *) row + dim*sizeof(float));
            ((half2 *) dst)[0] = __float22half2_rn(make_float2(value.x, value.y));
            ((half2 *) dst)[1] = __float22half2_rn(make_float2(value.z, value.w));
        } break;
        case GGML_TYPE_BF16: {
            float4 value;
            get_dequantize_V<GGML_TYPE_BF16, float, 4>()(row, &value, dim);
            ((half2 *) dst)[0] = __float22half2_rn(make_float2(value.x, value.y));
            ((half2 *) dst)[1] = __float22half2_rn(make_float2(value.z, value.w));
        } break;
        case GGML_TYPE_F16:
            get_dequantize_V<GGML_TYPE_F16, half, 4>()(row, dst, dim);
            break;
        case GGML_TYPE_Q8_0:
            get_dequantize_V<GGML_TYPE_Q8_0, half, 4>()(row, dst, dim);
            break;
        case GGML_TYPE_Q5_1:
            get_dequantize_V<GGML_TYPE_Q5_1, half, 4>()(row, dst, dim);
            break;
        case GGML_TYPE_Q5_0:
            get_dequantize_V<GGML_TYPE_Q5_0, half, 4>()(row, dst, dim);
            break;
        case GGML_TYPE_Q4_1:
            get_dequantize_V<GGML_TYPE_Q4_1, half, 4>()(row, dst, dim);
            break;
        case GGML_TYPE_Q4_0:
            get_dequantize_V<GGML_TYPE_Q4_0, half, 4>()(row, dst, dim);
            break;
        default:
            NO_DEVICE_CODE;
    }
}

static __global__ void msa_sparse_attn_kv_outer_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ block_idx,
        const char * __restrict__ pos_cell,
        const char * __restrict__ q_pos,
        const char * __restrict__ mask,
        char * __restrict__ dst,
        const int n_q_heads,
        const int n_kv_heads,
        const int n_tokens,
        const int n_streams,
        const int n_kv,
        const int n_pos,
        const int n_selected,
        const int block_size,
        const float scale,
        const ggml_type type_k,
        const ggml_type type_v,
        const size_t q_nb1,
        const size_t q_nb2,
        const size_t q_nb3,
        const size_t k_nb1,
        const size_t k_nb2,
        const size_t k_nb3,
        const size_t v_nb1,
        const size_t v_nb2,
        const size_t v_nb3,
        const size_t idx_nb1,
        const size_t idx_nb2,
        const size_t idx_nb3,
        const size_t pos_nb1,
        const size_t q_pos_nb1,
        const size_t mask_nb1,
        const size_t mask_nb3,
        const size_t dst_nb1,
        const size_t dst_nb2,
        const size_t dst_nb3) {
    constexpr int D            = 128;
    constexpr int Q_TILE       = 2;
    constexpr int K_TILE       = 16;
    constexpr int MAX_GQA      = 16;
    constexpr int MAX_SELECTED = 32;
    constexpr int MAX_UNION    = Q_TILE*MAX_SELECTED;

    const int tid       = threadIdx.x;
    const int query0    = blockIdx.x*Q_TILE;
    const int kv_head   = blockIdx.y;
    const int stream    = blockIdx.z;
    const int gqa       = n_q_heads/n_kv_heads;
    const int n_q_rows  = Q_TILE*gqa;

    __shared__ half  q_shared[Q_TILE*MAX_GQA*D];
    __shared__ half  k_shared[K_TILE*D];
    __shared__ half  v_shared[K_TILE*D];
    __shared__ float out_shared[Q_TILE*MAX_GQA*D];
    __shared__ float max_shared[Q_TILE*MAX_GQA];
    __shared__ float sum_shared[Q_TILE*MAX_GQA];
    __shared__ float score_shared[Q_TILE*MAX_GQA*K_TILE];
    __shared__ float alpha_shared[Q_TILE*MAX_GQA];
    __shared__ float beta_shared[Q_TILE*MAX_GQA*K_TILE];
    __shared__ int   query_pos_shared[Q_TILE];
    __shared__ int   key_cell_shared[K_TILE];
    __shared__ int   key_pos_shared[K_TILE];
    __shared__ int   union_blocks[MAX_UNION];
    __shared__ unsigned char union_query_mask[MAX_UNION];
    __shared__ int   n_union;

    for (int i = tid; i < n_q_rows*D; i += blockDim.x) {
        const int qr    = i/D;
        const int dim   = i%D;
        const int qt    = qr/gqa;
        const int gh    = qr%gqa;
        const int token = query0 + qt;
        if (token < n_tokens) {
            const int q_head = kv_head*gqa + gh;
            const float value = *(const float *) (q + dim*sizeof(float) +
                    q_head*q_nb1 + token*q_nb2 + stream*q_nb3);
            q_shared[i] = __float2half(value);
        } else {
            q_shared[i] = __float2half(0.0f);
        }
        out_shared[i] = 0.0f;
    }
    for (int qr = tid; qr < n_q_rows; qr += blockDim.x) {
        max_shared[qr] = -INFINITY;
        sum_shared[qr] = 0.0f;
    }

    if (tid < Q_TILE) {
        const int token = query0 + tid;
        query_pos_shared[tid] = token < n_tokens
                ? *(const int32_t *) (q_pos + token*sizeof(int32_t) + stream*q_pos_nb1)
                : -1;
    }

    if (tid == 0) {
        n_union = 0;
        for (int qt = 0; qt < Q_TILE && query0 + qt < n_tokens; ++qt) {
            const char * idx_row = block_idx + kv_head*idx_nb1 + (query0 + qt)*idx_nb2 + stream*idx_nb3;
            for (int rank = 0; rank < n_selected; ++rank) {
                const int block = *(const int32_t *) (idx_row + rank*sizeof(int32_t));
                if (block < 0 || (int64_t) block*block_size >= n_pos) {
                    continue;
                }
                int u = 0;
                for (; u < n_union && union_blocks[u] != block; ++u) {
                }
                if (u == n_union) {
                    union_blocks[u] = block;
                    union_query_mask[u] = 0;
                    ++n_union;
                }
                union_query_mask[u] |= 1u << qt;
            }
        }
    }
    __syncthreads();

    for (int u = 0; u < n_union; ++u) {
        const int pos0 = (int) ((int64_t) union_blocks[u]*block_size);
        const unsigned char query_mask = union_query_mask[u];

        for (int off = 0; off < block_size; off += K_TILE) {
            if (tid < K_TILE) {
                const int pos = pos0 + off + tid;
                const int cell = pos < n_pos
                        ? *(const int32_t *) (pos_cell + pos*sizeof(int32_t) + stream*pos_nb1)
                        : -1;
                key_pos_shared[tid]  = pos;
                key_cell_shared[tid] = cell >= 0 && cell < n_kv ? cell : -1;
            }
            __syncthreads();

            for (int i = tid; i < K_TILE*(D/4); i += blockDim.x) {
                const int kt   = i/(D/4);
                const int d4   = 4*(i%(D/4));
                const int cell = key_cell_shared[kt];
                if (cell >= 0) {
                    const void * k_row = k + kv_head*k_nb1 + cell*k_nb2 + stream*k_nb3;
                    const void * v_row = v + kv_head*v_nb1 + cell*v_nb2 + stream*v_nb3;
                    msa_dequantize_4(type_k, k_row, &k_shared[kt*D + d4], d4);
                    msa_dequantize_4(type_v, v_row, &v_shared[kt*D + d4], d4);
                } else {
                    *(int2 *) &k_shared[kt*D + d4] = make_int2(0, 0);
                    *(int2 *) &v_shared[kt*D + d4] = make_int2(0, 0);
                }
            }
            __syncthreads();

            if (tid < 128) {
                const int qr   = tid/4;
                const int lane = tid%4;
                const int qt   = qr/gqa;
                for (int kt = 0; kt < K_TILE; ++kt) {
                    float mask_value = -INFINITY;
                    if (qr < n_q_rows && query0 + qt < n_tokens && key_cell_shared[kt] >= 0) {
                        mask_value = __half2float(*(const half *) (mask + key_cell_shared[kt]*sizeof(half) +
                                (query0 + qt)*mask_nb1 + stream*mask_nb3));
                    }
                    const bool active = isfinite(mask_value) &&
                            (query_mask & (1u << qt)) && key_cell_shared[kt] >= 0 &&
                            key_pos_shared[kt] <= query_pos_shared[qt];
                    float dot = 0.0f;
                    if (active) {
                        for (int dim = 2*lane; dim < D; dim += 8) {
                            const float2 qv = __half22float2(*(const half2 *) &q_shared[qr*D + dim]);
                            const float2 kv = __half22float2(*(const half2 *) &k_shared[kt*D + dim]);
                            dot += qv.x*kv.x + qv.y*kv.y;
                        }
                    }
                    dot += __shfl_xor_sync(0xffffffff, dot, 1, 4);
                    dot += __shfl_xor_sync(0xffffffff, dot, 2, 4);
                    if (lane == 0 && qr < n_q_rows) {
                        score_shared[qr*K_TILE + kt] = active ? dot*scale + mask_value : -INFINITY;
                    }
                }
            }
            __syncthreads();

            if (tid < n_q_rows) {
                float tile_max = -INFINITY;
#pragma unroll
                for (int kt = 0; kt < K_TILE; ++kt) {
                    tile_max = fmaxf(tile_max, score_shared[tid*K_TILE + kt]);
                }
                const float old_max = max_shared[tid];
                const float new_max = fmaxf(old_max, tile_max);
                const float alpha = isfinite(old_max) ? expf(old_max - new_max) : 0.0f;
                float tile_sum = 0.0f;
#pragma unroll
                for (int kt = 0; kt < K_TILE; ++kt) {
                    const float score = score_shared[tid*K_TILE + kt];
                    const float beta = isfinite(score) ? expf(score - new_max) : 0.0f;
                    beta_shared[tid*K_TILE + kt] = beta;
                    tile_sum += beta;
                }
                alpha_shared[tid] = isfinite(new_max) ? alpha : 1.0f;
                sum_shared[tid]   = alpha*sum_shared[tid] + tile_sum;
                max_shared[tid]   = new_max;
            }
            __syncthreads();

            for (int i = tid; i < n_q_rows*D; i += blockDim.x) {
                const int qr  = i/D;
                const int dim = i%D;
                float value = 0.0f;
#pragma unroll
                for (int kt = 0; kt < K_TILE; ++kt) {
                    value += beta_shared[qr*K_TILE + kt]*__half2float(v_shared[kt*D + dim]);
                }
                out_shared[i] = alpha_shared[qr]*out_shared[i] + value;
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < n_q_rows*D; i += blockDim.x) {
        const int qr    = i/D;
        const int dim   = i%D;
        const int qt    = qr/gqa;
        const int gh    = qr%gqa;
        const int token = query0 + qt;
        if (token < n_tokens) {
            const int q_head = kv_head*gqa + gh;
            const float sum = sum_shared[qr];
            *(float *) (dst + dim*sizeof(float) + q_head*dst_nb1 + token*dst_nb2 + stream*dst_nb3) =
                    sum > 0.0f ? out_shared[i]/sum : 0.0f;
        }
    }
}

#endif

void ggml_cuda_op_msa_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const ggml_tensor * q         = dst->src[0];
    const ggml_tensor * k         = dst->src[1];
    const ggml_tensor * v         = dst->src[2];
    const ggml_tensor * block_idx = dst->src[3];
    const ggml_tensor * pos_cell  = dst->src[4];
    const ggml_tensor * q_pos     = dst->src[5];
    const ggml_tensor * mask      = dst->src[6];

    constexpr int Q_TILE = 2;
    const dim3 block(256);
    const dim3 grid((q->ne[2] + Q_TILE - 1)/Q_TILE, k->ne[1], q->ne[3]);

    msa_sparse_attn_kv_outer_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const char *) block_idx->data, (const char *) pos_cell->data, (const char *) q_pos->data,
            (const char *) mask->data, (char *) dst->data,
            q->ne[1], k->ne[1], q->ne[2], q->ne[3], k->ne[2], pos_cell->ne[0],
            block_idx->ne[0], ggml_get_op_params_i32(dst, 0), ggml_get_op_params_f32(dst, 1),
            k->type, v->type, q->nb[1], q->nb[2], q->nb[3], k->nb[1], k->nb[2], k->nb[3],
            v->nb[1], v->nb[2], v->nb[3], block_idx->nb[1], block_idx->nb[2], block_idx->nb[3],
            pos_cell->nb[1], q_pos->nb[1], mask->nb[1], mask->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("MSA sparse attention is only implemented for NVIDIA CUDA");
#endif
}

static bool msa_sparse_attn_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_msa_sparse_attn_supported(const ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const ggml_tensor * q         = dst->src[0];
    const ggml_tensor * k         = dst->src[1];
    const ggml_tensor * v         = dst->src[2];
    const ggml_tensor * block_idx = dst->src[3];
    const ggml_tensor * pos_cell  = dst->src[4];
    const ggml_tensor * q_pos     = dst->src[5];
    const ggml_tensor * mask      = dst->src[6];

    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || q->ne[0] != 128 ||
            k->ne[0] != 128 || v->ne[0] != 128 || k->ne[1] <= 0 || k->ne[1] != v->ne[1] ||
            q->ne[1] % k->ne[1] != 0 || q->ne[1]/k->ne[1] > 16 || k->ne[2] != v->ne[2] ||
            q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] || !ggml_are_same_shape(q, dst)) {
        return false;
    }
    if (q->ne[1] > INT_MAX || q->ne[2] > INT_MAX || q->ne[3] > INT_MAX ||
            k->ne[1] > INT_MAX || k->ne[2] > INT_MAX || pos_cell->ne[0] > INT_MAX) {
        return false;
    }
    if (block_idx->type != GGML_TYPE_I32 || block_idx->ne[0] <= 0 || block_idx->ne[0] > 32 ||
            block_idx->ne[1] != k->ne[1] || block_idx->ne[2] != q->ne[2] || block_idx->ne[3] != q->ne[3] ||
            pos_cell->type != GGML_TYPE_I32 || pos_cell->ne[1] != q->ne[3] ||
            pos_cell->ne[2] != 1 || pos_cell->ne[3] != 1 ||
            q_pos->type != GGML_TYPE_I32 || q_pos->ne[0] != q->ne[2] || q_pos->ne[1] != q->ne[3] ||
            q_pos->ne[2] != 1 || q_pos->ne[3] != 1) {
        return false;
    }
    const int block_size = ggml_get_op_params_i32(dst, 0);
    if (block_size != 128 || pos_cell->ne[0] % block_size != 0 ||
            block_idx->ne[0] > pos_cell->ne[0]/block_size || mask->type != GGML_TYPE_F16 ||
            mask->ne[0] != k->ne[2] || mask->ne[1] != q->ne[2] || mask->ne[2] != 1 || mask->ne[3] != q->ne[3]) {
        return false;
    }
    return msa_sparse_attn_cache_type_supported(k->type) && msa_sparse_attn_cache_type_supported(v->type) &&
            mask->nb[0] == sizeof(half) &&
            ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(k) && ggml_is_contiguous_rows(v) &&
            ggml_is_contiguous(block_idx) && ggml_is_contiguous(pos_cell) && ggml_is_contiguous(q_pos) &&
            ggml_is_contiguous_rows(mask) && ggml_is_contiguous(dst);
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

static __global__ void msa_block_mask_kernel(
        const int * __restrict__ idx,
        const char * __restrict__ mask,
        half * __restrict__ dst,
        const int n_kv,
        const int n_tokens,
        const int n_heads,
        const int n_blocks,
        const int block_size,
        const int k,
        const size_t mask_nb1) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    const int * idx_row = idx + k * (head + n_heads * token);
    const half * mask_row = (const half *) (mask + token * mask_nb1);
    half * dst_row = dst + n_kv * (token + n_tokens * head);

    for (int block = warp; block < n_blocks; block += warps) {
        bool selected = false;
        for (int i = lane; i < k; i += 32) {
            selected |= idx_row[i] == block;
        }
        selected = __any_sync(0xffffffff, selected);

        const int first = block * block_size;
        for (int i = lane; i < block_size && first + i < n_kv; i += 32) {
            dst_row[first + i] = selected ? mask_row[first + i] : __float2half(-INFINITY);
        }
    }
}

void ggml_cuda_op_msa_block_mask(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *        idx,
        const ggml_tensor *        mask,
        ggml_tensor *              dst) {
    GGML_ASSERT(idx->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(idx));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(mask->nb[0] == sizeof(half));

    const int k        = idx->ne[0];
    const int n_heads  = idx->ne[1];
    const int n_tokens = idx->ne[2];
    const int n_blocks = idx->src[0]->ne[0];
    const int n_kv     = dst->ne[0];

    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_heads);
    GGML_ASSERT(mask->ne[0] == n_kv);
    GGML_ASSERT(mask->ne[1] == n_tokens);
    GGML_ASSERT(n_kv % n_blocks == 0);

    const int block_size = n_kv / n_blocks;
    const dim3 blocks(n_tokens, n_heads);
    const int threads = 256;

    msa_block_mask_kernel<<<blocks, threads, 0, ctx.stream()>>>(
            (const int *) idx->data, (const char *) mask->data, (half *) dst->data,
            n_kv, n_tokens, n_heads, n_blocks, block_size, k, mask->nb[1]);
}

static __global__ void msa_block_mask_mapped_kernel(
        const char * __restrict__ idx,
        const int * __restrict__ cell_block,
        const char * __restrict__ mask,
        half * __restrict__ dst,
        const int n_kv,
        const int n_tokens,
        const int n_heads,
        const int k,
        const size_t idx_nb1,
        const size_t idx_nb2,
        const size_t mask_nb1) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;

    const int * idx_row = (const int *) (idx + head * idx_nb1 + token * idx_nb2);
    const half * mask_row = (const half *) (mask + token * mask_nb1);
    half * dst_row = dst + n_kv * (token + n_tokens * head);

    for (int cell = threadIdx.x; cell < n_kv; cell += blockDim.x) {
        const int block = cell_block[cell];
        bool selected = false;
        for (int i = 0; i < k; ++i) {
            selected |= idx_row[i] == block;
        }
        dst_row[cell] = selected ? mask_row[cell] : __float2half(-INFINITY);
    }
}

void ggml_cuda_op_msa_block_mask_mapped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *        idx,
        const ggml_tensor *        cell_block,
        const ggml_tensor *        mask,
        ggml_tensor *              dst) {
    GGML_ASSERT(idx->type == GGML_TYPE_I32);
    GGML_ASSERT(cell_block->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(cell_block));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(mask->nb[0] == sizeof(half));

    const int k        = idx->ne[0];
    const int n_heads  = idx->ne[1];
    const int n_tokens = idx->ne[2];
    const int n_kv     = dst->ne[0];

    GGML_ASSERT(cell_block->ne[0] == n_kv);
    GGML_ASSERT(cell_block->ne[1] == 1 && cell_block->ne[2] == 1 && cell_block->ne[3] == 1);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_heads);
    GGML_ASSERT(mask->ne[0] == n_kv);
    GGML_ASSERT(mask->ne[1] == n_tokens);
    GGML_ASSERT(mask->ne[2] == 1 && mask->ne[3] == 1);

    const dim3 blocks(n_tokens, n_heads);
    const int threads = 256;

    msa_block_mask_mapped_kernel<<<blocks, threads, 0, ctx.stream()>>>(
            (const char *) idx->data, (const int *) cell_block->data,
            (const char *) mask->data, (half *) dst->data,
            n_kv, n_tokens, n_heads, k, idx->nb[1], idx->nb[2], mask->nb[1]);
}
