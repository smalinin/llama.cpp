#include "dsv41-attn-indices.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cstdint>

template<int ITEMS>
__global__ void dsv41_prefill_indices(
        const half * raw_mask, const half * comp_mask, const int32_t * top,
        int32_t * output, int32_t * counts, int nr, int nc, int k, int stride, int bits) {
    constexpr int THREADS = 256;
    using Sort = cub::BlockRadixSort<uint32_t, THREADS, ITEMS>;
    using Scan = cub::BlockScan<int, THREADS>;
    __shared__ union { typename Sort::TempStorage sort; typename Scan::TempStorage scan; } scratch;
    __shared__ uint32_t sorted[THREADS*ITEMS];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t values[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = tid*ITEMS + j;
        uint32_t value = UINT32_MAX;
        if (i < nr) {
            if (isfinite(__half2float(raw_mask[int64_t(row)*nr + i]))) value = i;
        } else if (i < nr + k) {
            const int id = top[int64_t(row)*k + i - nr];
            if (id >= 0 && id < nc && isfinite(__half2float(comp_mask[int64_t(row)*nc + id]))) value = nr + id;
        }
        values[j] = value;
    }
    Sort(scratch.sort).Sort(values, 0, bits);
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) sorted[tid*ITEMS + j] = values[j];
    for (int i = tid; i < stride; i += THREADS) output[int64_t(row)*stride + i] = -1;
    __syncthreads();
    int local_count = 0;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = tid*ITEMS + j;
        local_count += values[j] != UINT32_MAX && (i == 0 || values[j] != sorted[i-1]);
    }
    int offset, total;
    Scan(scratch.scan).ExclusiveSum(local_count, offset, total);
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int i = tid*ITEMS + j;
        if (values[j] != UINT32_MAX && (i == 0 || values[j] != sorted[i-1])) {
            if (offset < stride) output[int64_t(row)*stride + offset] = int32_t(values[j]);
            ++offset;
        }
    }
    if (tid == 0) counts[row] = total;
}

static void launch_indices(const void * raw_mask, const void * comp_mask, const void * top,
        void * output, void * counts, int nr, int nc, int k, int nq, int window, cudaStream_t stream) {
    int bits = 1;
    while ((1 << bits) <= nc + nr) ++bits;
    const auto raw = static_cast<const half *>(raw_mask);
    const auto comp = static_cast<const half *>(comp_mask);
    const auto ids = static_cast<const int32_t *>(top);
    auto out = static_cast<int32_t *>(output);
    auto count = static_cast<int32_t *>(counts);
    const int stride = k + window;
    if (nr + k <= 1024)
        dsv41_prefill_indices<4><<<nq, 256, 0, stream>>>(raw, comp, ids, out, count, nr, nc, k, stride, bits);
    else if (nr + k <= 2048)
        dsv41_prefill_indices<8><<<nq, 256, 0, stream>>>(raw, comp, ids, out, count, nr, nc, k, stride, bits);
    else
        dsv41_prefill_indices<12><<<nq, 256, 0, stream>>>(raw, comp, ids, out, count, nr, nc, k, stride, bits);
    CUDA_CHECK(cudaGetLastError());
}
#endif

bool ggml_cuda_dsv41_attn_indices_supported(int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const int cc = ggml_cuda_info().devices[device].cc;
    const auto * raw = dst->src[0];
    const auto * comp = dst->src[1];
    const auto * top = dst->src[2];
    const int window = ggml_get_op_params_i32(dst, 0);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_HOPPER &&
        raw && comp && top && dst->type == GGML_TYPE_I32 && raw->type == GGML_TYPE_F16 &&
        comp->type == GGML_TYPE_F16 && top->type == GGML_TYPE_I32 && window == 128 &&
        ggml_is_matrix(raw) && ggml_is_matrix(comp) && ggml_is_matrix(top) &&
        ggml_is_contiguous(raw) && ggml_is_contiguous(comp) && ggml_is_contiguous(top) && ggml_is_contiguous(dst) &&
        raw->ne[0] > 0 && raw->ne[0] <= 1024 && comp->ne[0] >= top->ne[0] &&
        raw->ne[0] + comp->ne[0] < (1LL << 24) && top->ne[1] > 0 && top->ne[1] <= 512 &&
        raw->ne[1] >= top->ne[1] && comp->ne[1] >= top->ne[1] &&
        (top->ne[0] == 512 || top->ne[0] == 768 || top->ne[0] == 1024 || top->ne[0] == 2048) &&
        ggml_nelements(dst) == (top->ne[0] + window + 1)*top->ne[1];
#endif
}

void ggml_cuda_dsv41_attn_indices(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv41_attn_indices_supported(ctx.device, dst));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const auto * raw = dst->src[0];
    const auto * comp = dst->src[1];
    const auto * top = dst->src[2];
    const int window = ggml_get_op_params_i32(dst, 0);
    auto * output = (int32_t *) dst->data;
    launch_indices(raw->data, comp->data, top->data, output, output + (top->ne[0] + window)*top->ne[1],
        raw->ne[0], comp->ne[0], top->ne[0], top->ne[1], window, ctx.stream());
#endif
}
