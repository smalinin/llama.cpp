#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.2
#endif      // GGML_CUDA_USE_CUB

#ifdef GGML_CUDA_USE_CUB

// Map finite floats to unsigned keys with the same order. Keep NaNs below -infinity.
static __device__ __forceinline__ uint32_t top_k_float_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return 1;
    }
    return bits ^ ((int32_t(bits) < 0) ? 0xffffffffU : 0x80000000U);
}

struct top_k_radix_state {
    uint32_t prefix;
    int rank;
    int n_greater;
    int n_equal;
};

// Find the exact kth key one byte at a time, then collect greater keys and enough ties.
static __global__ void top_k_radix_init(uint32_t * histogram, top_k_radix_state * state, int k) {
    histogram += blockIdx.x * 256;
    state += blockIdx.x;
    histogram[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
        state->prefix = 0;
        state->rank = k;
        state->n_greater = 0;
        state->n_equal = 0;
    }
}

template<int shift>
static __global__ void top_k_radix_histogram(
        const float * src, int ncols, const top_k_radix_state * state, uint32_t * histogram) {
    __shared__ uint32_t local_histogram[256];
    local_histogram[threadIdx.x] = 0;
    __syncthreads();

    const int row = blockIdx.y;
    src += (size_t) row * ncols;
    state += row;
    histogram += row * 256;

    const uint32_t prefix = state->prefix;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols; col += blockDim.x * gridDim.x) {
        const uint32_t key = top_k_float_key(src[col]);
        bool matches_prefix = true;
        if constexpr (shift != 24) {
            matches_prefix = (key >> (shift + 8)) == (prefix >> (shift + 8));
        }
        if (matches_prefix) {
            atomicAdd(&local_histogram[(key >> shift) & 0xff], 1);
        }
    }
    __syncthreads();

    if (local_histogram[threadIdx.x] != 0) {
        atomicAdd(&histogram[threadIdx.x], local_histogram[threadIdx.x]);
    }
}

template<int shift>
static __global__ void top_k_radix_select(uint32_t * histogram, top_k_radix_state * state) {
    using block_scan = cub::BlockScan<int, 256>;
    __shared__ typename block_scan::TempStorage temp_storage;
    __shared__ int selected_digit;
    __shared__ int selected_rank;

    histogram += blockIdx.x * 256;
    state += blockIdx.x;

    const int digit = 255 - threadIdx.x;
    const int count = histogram[digit];
    int inclusive_count;
    block_scan(temp_storage).InclusiveSum(count, inclusive_count);

    const int rank = state->rank;
    const int preceding_count = inclusive_count - count;
    if (preceding_count < rank && rank <= inclusive_count) {
        selected_digit = digit;
        selected_rank = rank - preceding_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        state->prefix |= uint32_t(selected_digit) << shift;
        state->rank = selected_rank;
    }
    histogram[threadIdx.x] = 0;
}

static __device__ __forceinline__ void top_k_radix_emit(
        bool selected, int col, int * counter, int limit, int offset, int * dst) {
    const unsigned int active = __activemask();
    const unsigned int selected_mask = __ballot_sync(active, selected);
    if (selected_mask == 0) {
        return;
    }

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int leader = __ffs(selected_mask) - 1;
    int base = 0;
    if (lane == leader) {
        base = atomicAdd(counter, __popc(selected_mask));
    }
    base = __shfl_sync(active, base, leader);

    const int pos = base + __popc(selected_mask & ((1U << lane) - 1));
    if (selected && pos < limit) {
        dst[offset + pos] = col;
    }
}

static __global__ void top_k_radix_collect(
        const float * src, int * dst, int ncols, int k, top_k_radix_state * state) {
    const int row = blockIdx.y;
    src += (size_t) row * ncols;
    dst += (size_t) row * k;
    state += row;

    const uint32_t threshold = state->prefix;
    const int n_greater = k - state->rank;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols; col += blockDim.x * gridDim.x) {
        const uint32_t key = top_k_float_key(src[col]);
        top_k_radix_emit(key > threshold, col, &state->n_greater, n_greater, 0, dst);
        top_k_radix_emit(key == threshold, col, &state->n_equal, state->rank, n_greater, dst);
    }
}

static void top_k_partial_cub(
        ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int block_size = 256;
    const int nblocks = std::min(128, (ncols + block_size - 1) / block_size);

    ggml_cuda_pool_alloc<uint32_t> histogram_alloc(pool, (size_t) 256 * nrows);
    ggml_cuda_pool_alloc<top_k_radix_state> state_alloc(pool, nrows);
    uint32_t * histogram = histogram_alloc.get();
    top_k_radix_state * state = state_alloc.get();

    const dim3 histogram_grid(nblocks, nrows, 1);
    top_k_radix_init<<<nrows, block_size, 0, stream>>>(histogram, state, k);
    top_k_radix_histogram<24><<<histogram_grid, block_size, 0, stream>>>(src, ncols, state, histogram);
    top_k_radix_select<24><<<nrows, block_size, 0, stream>>>(histogram, state);
    top_k_radix_histogram<16><<<histogram_grid, block_size, 0, stream>>>(src, ncols, state, histogram);
    top_k_radix_select<16><<<nrows, block_size, 0, stream>>>(histogram, state);
    top_k_radix_histogram< 8><<<histogram_grid, block_size, 0, stream>>>(src, ncols, state, histogram);
    top_k_radix_select< 8><<<nrows, block_size, 0, stream>>>(histogram, state);
    top_k_radix_histogram< 0><<<histogram_grid, block_size, 0, stream>>>(src, ncols, state, histogram);
    top_k_radix_select< 0><<<nrows, block_size, 0, stream>>>(histogram, state);
    top_k_radix_collect<<<histogram_grid, block_size, 0, stream>>>(src, dst, ncols, k, state);
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void top_k_bitonic(ggml_cuda_pool & pool,
                          const float *    src,
                          int *            dst,
                          const int        ncols,
                          const int        nrows,
                          const int        k,
                          cudaStream_t     stream) {
    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int * tmp_dst = temp_dst_alloc.get();

    for (int i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        argsort_f32_i32_cuda_bitonic(src, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                    cudaMemcpyDeviceToDevice, stream));

        src += ncols * iter_nrows;
        dst += k     * iter_nrows;
    }
}

static void top_k_argsort_cub(ggml_cuda_pool & pool,
                              const float *    src,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              const int        k,
                              cudaStream_t     stream) {
    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(ncols * sizeof(float), nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int * tmp_dst = temp_dst_alloc.get();

    for (int i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        argsort_f32_i32_cuda_cub(pool, src, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                    cudaMemcpyDeviceToDevice, stream));

        src += ncols * iter_nrows;
        dst += k     * iter_nrows;
    }
}

#endif  // GGML_CUDA_USE_CUB

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

#endif  // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
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
#ifdef CUB_TOP_K_AVAILABLE
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    if (nrows > 1) {
        if (ncols <= 1024 && shared_mem <= max_shared_mem) {
            top_k_bitonic(pool, src0_d, dst_d, ncols, nrows, k, stream);
        } else {
            // DeviceTopK only supports one row. A single segmented argsort is
            // faster than launching DeviceTopK sequentially for every token.
            top_k_argsort_cub(pool, src0_d, dst_d, ncols, nrows, k, stream);
        }
        return;
    }

    top_k_cub(pool, src0_d, dst_d, ncols, k, stream);
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (nrows <= 65535 && ncols >= 4096 && ncols <= 1024*1024 && k <= 512) {
        top_k_partial_cub(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;

    if (use_bitonic) {
        top_k_bitonic(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    top_k_argsort_cub(pool, src0_d, dst_d, ncols, nrows, k, stream);
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
