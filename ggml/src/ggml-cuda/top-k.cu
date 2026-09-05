#include "argsort.cuh"
#include "top-k.cuh"

#include <cstdlib>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Exact, unsorted Top-K for the wide-selection case used by the GLM lightning
// indexer. Older CCCL releases do not have DeviceTopK and otherwise fall back
// to sorting every score. Four radix passes find the Kth key, then deterministic
// scans emit every larger key and enough ties to produce exactly K indices.
static __device__ __forceinline__ uint32_t top_k_ordered_f32(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}

struct top_k_radix_shared {
    uint32_t histogram[256];
    uint32_t prefix;
    uint32_t prefix_mask;
    uint32_t threshold;
    int rank;
    int n_greater;
    int warp_greater[32];
    int warp_equal[32];
};

template<int BLOCK_SIZE>
static __device__ void top_k_emit_selected(
        const float * src, int * dst, int ncols, int k, top_k_radix_shared & shared) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int nwarps = BLOCK_SIZE/32;
    static_assert(BLOCK_SIZE % 32 == 0, "Top-K block size must contain whole warps");

    int local_greater = 0;
    int local_equal = 0;
    for (int col = tid; col < ncols; col += BLOCK_SIZE) {
        const uint32_t key = top_k_ordered_f32(src[col]);
        local_greater += key > shared.threshold;
        local_equal += key == shared.threshold;
    }

    int greater_scan = local_greater;
    int equal_scan = local_equal;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int greater_up = __shfl_up_sync(0xffffffffu, greater_scan, offset);
        const int equal_up = __shfl_up_sync(0xffffffffu, equal_scan, offset);
        if (lane >= offset) {
            greater_scan += greater_up;
            equal_scan += equal_up;
        }
    }

    if (lane == 31) {
        shared.warp_greater[warp] = greater_scan;
        shared.warp_equal[warp] = equal_scan;
    }
    __syncthreads();

    if (warp == 0) {
        int warp_greater = lane < nwarps ? shared.warp_greater[lane] : 0;
        int warp_equal = lane < nwarps ? shared.warp_equal[lane] : 0;
        const int own_greater = warp_greater;
        const int own_equal = warp_equal;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int greater_up = __shfl_up_sync(0xffffffffu, warp_greater, offset);
            const int equal_up = __shfl_up_sync(0xffffffffu, warp_equal, offset);
            if (lane >= offset) {
                warp_greater += greater_up;
                warp_equal += equal_up;
            }
        }
        if (lane < nwarps) {
            shared.warp_greater[lane] = warp_greater - own_greater;
            shared.warp_equal[lane] = warp_equal - own_equal;
        }
        if (lane == nwarps - 1) {
            shared.n_greater = warp_greater;
        }
    }
    __syncthreads();

    int out_greater = shared.warp_greater[warp] + greater_scan - local_greater;
    int out_equal = shared.warp_equal[warp] + equal_scan - local_equal;
    const int n_equal_wanted = k - shared.n_greater;
    for (int col = tid; col < ncols; col += BLOCK_SIZE) {
        const uint32_t key = top_k_ordered_f32(src[col]);
        if (key > shared.threshold) {
            dst[out_greater++] = col;
        } else if (key == shared.threshold) {
            if (out_equal < n_equal_wanted) {
                dst[shared.n_greater + out_equal] = col;
            }
            out_equal++;
        }
    }
}

template<int BLOCK_SIZE>
static __device__ void top_k_radix_select_row(
        const float * src, int * dst, int ncols, int k, top_k_radix_shared & shared) {
    const int tid = threadIdx.x;

    if (tid == 0) {
        shared.prefix      = 0;
        shared.prefix_mask = 0;
        shared.rank        = k - 1;
    }
    __syncthreads();

#pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        shared.histogram[tid] = 0;
        __syncthreads();

        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            const uint32_t key = top_k_ordered_f32(src[col]);
            if ((key & shared.prefix_mask) == shared.prefix) {
                atomicAdd(&shared.histogram[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int rank_cur = shared.rank;
            int digit = 255;
            for (; digit >= 0; --digit) {
                const int count = shared.histogram[digit];
                if (rank_cur < count) {
                    break;
                }
                rank_cur -= count;
            }
            shared.prefix      |= (uint32_t) digit << shift;
            shared.prefix_mask |= 0xffu << shift;
            shared.rank         = rank_cur;
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared.threshold = shared.prefix;
    }
    __syncthreads();

    top_k_emit_selected<BLOCK_SIZE>(src, dst, ncols, k, shared);
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_select_f32_i32(
        const float * src, int * dst, int ncols, int k) {
    const int row = blockIdx.x;

    src += (size_t) row*ncols;
    dst += (size_t) row*k;

    __shared__ top_k_radix_shared shared;
    top_k_radix_select_row<BLOCK_SIZE>(src, dst, ncols, k, shared);
}

template<int BLOCK_SIZE>
static __device__ int top_k_count_ge(
        const float * src, int ncols, uint32_t threshold, int * count) {
    if (threadIdx.x == 0) {
        *count = 0;
    }
    __syncthreads();

    int local = 0;
    for (int col = threadIdx.x; col < ncols; col += BLOCK_SIZE) {
        local += top_k_ordered_f32(src[col]) >= threshold;
    }
    if (local != 0) {
        atomicAdd(count, local);
    }
    __syncthreads();

    return *count;
}

template<int BLOCK_SIZE, int MAX_CANDIDATES>
static __global__ void top_k_temporal_f32_i32(
        const float * src, const int * hint, int * dst, int ncols, int k) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    src  += (size_t) row*ncols;
    hint += (size_t) row*k;
    dst  += (size_t) row*k;

    __shared__ float reduce_min[BLOCK_SIZE];
    __shared__ float reduce_max[BLOCK_SIZE];
    __shared__ float reduce_sum[BLOCK_SIZE];
    __shared__ int reduce_count[BLOCK_SIZE];
    __shared__ int candidate_count;
    __shared__ int sort_n;
    __shared__ uint32_t candidate_keys[MAX_CANDIDATES];
    __shared__ top_k_radix_shared radix;

    float local_min = INFINITY;
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    int local_count = 0;

    for (int i = tid; i < k; i += BLOCK_SIZE) {
        const int col = hint[i];
        if ((uint32_t) col < (uint32_t) ncols) {
            const float value = src[col];
            if (isfinite(value)) {
                local_min = fminf(local_min, value);
                local_max = fmaxf(local_max, value);
                local_sum += value;
                local_count++;
            }
        }
    }

    reduce_min[tid] = local_min;
    reduce_max[tid] = local_max;
    reduce_sum[tid] = local_sum;
    reduce_count[tid] = local_count;
    __syncthreads();

    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_min[tid] = fminf(reduce_min[tid], reduce_min[tid + stride]);
            reduce_max[tid] = fmaxf(reduce_max[tid], reduce_max[tid + stride]);
            reduce_sum[tid] += reduce_sum[tid + stride];
            reduce_count[tid] += reduce_count[tid + stride];
        }
        __syncthreads();
    }

    bool use_candidates = false;
    uint32_t threshold = 0;

    if (reduce_count[0] == k) {
        const float pmin = reduce_min[0];
        const float pmax = reduce_max[0];
        float probe = fminf(pmax, fmaxf(pmin, reduce_sum[0]/k));
        int count = top_k_count_ge<BLOCK_SIZE>(src, ncols, top_k_ordered_f32(probe), &candidate_count);

        if (count >= k && count <= MAX_CANDIDATES) {
            threshold = top_k_ordered_f32(probe);
            use_candidates = true;
        } else if (pmax > pmin) {
            float low;
            float high;
            int count_low;
            int count_high;

            if (count < k) {
                low = pmin;
                count_low = top_k_count_ge<BLOCK_SIZE>(src, ncols, top_k_ordered_f32(low), &candidate_count);
                high = probe;
                count_high = count;
            } else {
                low = probe;
                count_low = count;
                high = pmax;
                count_high = top_k_count_ge<BLOCK_SIZE>(src, ncols, top_k_ordered_f32(high), &candidate_count);
            }

            if (count_low >= k && count_low <= MAX_CANDIDATES) {
                threshold = top_k_ordered_f32(low);
                use_candidates = true;
            } else if (count_high >= k && count_high <= MAX_CANDIDATES) {
                threshold = top_k_ordered_f32(high);
                use_candidates = true;
            } else if (count_low > MAX_CANDIDATES && count_high < k) {
                const int target = k + (MAX_CANDIDATES - k)/4;

#pragma unroll
                for (int iter = 0; iter < 3 && !use_candidates; ++iter) {
                    const float fraction = float(count_low - target)/float(count_low - count_high);
                    const float next = low + (high - low)*fraction;
                    if (!(next > low && next < high)) {
                        break;
                    }

                    count = top_k_count_ge<BLOCK_SIZE>(src, ncols, top_k_ordered_f32(next), &candidate_count);
                    if (count >= k && count <= MAX_CANDIDATES) {
                        threshold = top_k_ordered_f32(next);
                        use_candidates = true;
                    } else if (count > MAX_CANDIDATES) {
                        low = next;
                        count_low = count;
                    } else {
                        high = next;
                        count_high = count;
                    }
                }
            }
        }
    }

    if (use_candidates) {
        if (tid == 0) {
            candidate_count = 0;
        }
        __syncthreads();

        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            const uint32_t key = top_k_ordered_f32(src[col]);
            if (key >= threshold) {
                const int pos = atomicAdd(&candidate_count, 1);
                if (pos < MAX_CANDIDATES) {
                    candidate_keys[pos] = key;
                }
            }
        }
        __syncthreads();

        if (candidate_count >= k && candidate_count <= MAX_CANDIDATES) {
            if (tid == 0) {
                int value = 1;
                while (value < candidate_count) {
                    value <<= 1;
                }
                sort_n = value;
            }
            __syncthreads();

            for (int i = candidate_count + tid; i < sort_n; i += BLOCK_SIZE) {
                candidate_keys[i] = 0;
            }
            __syncthreads();

            for (int size = 2; size <= sort_n; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    for (int i = tid; i < sort_n; i += BLOCK_SIZE) {
                        const int other = i ^ stride;
                        if (other > i) {
                            const uint32_t key_i = candidate_keys[i];
                            const uint32_t key_o = candidate_keys[other];
                            const bool ascending = (i & size) == 0;
                            if ((key_i > key_o) == ascending) {
                                candidate_keys[i] = key_o;
                                candidate_keys[other] = key_i;
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            if (tid == 0) {
                radix.threshold = candidate_keys[sort_n - k];
            }
            __syncthreads();

            top_k_emit_selected<BLOCK_SIZE>(src, dst, ncols, k, radix);
            return;
        }
    }

    top_k_radix_select_row<BLOCK_SIZE>(src, dst, ncols, k, radix);
}

static void top_k_radix_select_nvidia(
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int block_size = 256;
    top_k_radix_select_f32_i32<block_size><<<nrows, block_size, 0, stream>>>(src, dst, ncols, k);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (src1 != nullptr) {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(src1->ne[0] == k && ggml_nrows(src1) == nrows);

        const char * temporal_env = std::getenv("GGML_CUDA_TOPK_TEMPORAL");
        const bool use_temporal = temporal_env == nullptr || std::atoi(temporal_env) != 0;
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const bool supported_device = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE;
        // Below 24576 columns the exact radix selector is faster on Ampere/Ada.
        if (use_temporal && supported_device && ncols >= 24576 && ncols <= 49152 && k == 512) {
            constexpr int block_size = 256;
            constexpr int max_candidates = 2048;
            top_k_temporal_f32_i32<block_size, max_candidates><<<nrows, block_size, 0, stream>>>(
                    src0_d, (const int *) src1->data, dst_d, ncols, k);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if !defined(CUB_TOP_K_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // GLM selects 512 pools. With CCCL < 3.2 the generic fallback fully sorts
    // every active pool, even though Top-K explicitly does not require order.
    // Keep a switch for model-level A/B validation.
    const char * radix_env = std::getenv("GGML_CUDA_TOPK_RADIX_SELECT");
    const bool use_radix_select = radix_env == nullptr || std::atoi(radix_env) != 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool supported_device = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE;
    const int max_radix_cols = cc >= GGML_CUDA_CC_ADA_LOVELACE ? 49152 : 32768;
    if (use_radix_select && supported_device && ncols > 1024 && ncols <= max_radix_cols && k == 512) {
        top_k_radix_select_nvidia(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
#endif

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
