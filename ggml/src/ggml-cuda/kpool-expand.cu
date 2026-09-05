#include "kpool-expand.cuh"

static __global__ void kpool_expand_kernel(
        const int32_t * __restrict__ selected,
        const int32_t * __restrict__ pool_cells,
        const float   * __restrict__ pool_bias,
        const int32_t * __restrict__ tail_cells,
        const half    * __restrict__ tail_mask,
              int32_t * __restrict__ dst,
        int kpool,
        int n_select,
        int n_pools,
        int n_tail,
        int n_compact,
        int n_query,
        int64_t n_elements) {
    const int64_t index = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= n_elements) {
        return;
    }

    const int64_t row = index/n_compact;
    const int64_t stream = row/n_query;
    const int i = index % n_compact;

    if (i < n_select*kpool) {
        const int pool = selected[row*n_select + i/kpool];
        const bool valid = pool >= 0 && pool < n_pools && isfinite(pool_bias[row*n_pools + pool]);
        dst[index] = valid ? pool_cells[(stream*n_pools + pool)*kpool + i%kpool] : -1;
        return;
    }

    const int it = i - n_select*kpool;
    const bool valid = it < n_tail && isfinite(__half2float(tail_mask[row*n_tail + it]));
    dst[index] = valid ? tail_cells[row*n_tail + it] : -1;
}

bool ggml_cuda_kpool_expand_supported(const ggml_tensor * dst) {
    const ggml_tensor * selected   = dst->src[0];
    const ggml_tensor * pool_cells = dst->src[1];
    const ggml_tensor * pool_bias  = dst->src[2];
    const ggml_tensor * tail_cells = dst->src[3];
    const ggml_tensor * tail_mask  = dst->src[4];

    return dst->type == GGML_TYPE_I32 && selected->type == GGML_TYPE_I32 &&
        pool_cells->type == GGML_TYPE_I32 && pool_bias->type == GGML_TYPE_F32 &&
        tail_cells->type == GGML_TYPE_I32 && tail_mask->type == GGML_TYPE_F16 &&
        ggml_is_contiguous(dst) && ggml_is_contiguous(selected) &&
        ggml_is_contiguous(pool_cells) && ggml_is_contiguous(pool_bias) &&
        ggml_is_contiguous(tail_cells) && ggml_is_contiguous(tail_mask);
}

void ggml_cuda_kpool_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_kpool_expand_supported(dst));

    const ggml_tensor * selected   = dst->src[0];
    const ggml_tensor * pool_cells = dst->src[1];
    const ggml_tensor * pool_bias  = dst->src[2];
    const ggml_tensor * tail_cells = dst->src[3];
    const ggml_tensor * tail_mask  = dst->src[4];

    const int kpool = ggml_get_op_params_i32(dst, 0);
    constexpr int threads = 256;
    const int64_t n_elements = ggml_nelements(dst);
    const int blocks = (n_elements + threads - 1)/threads;

    kpool_expand_kernel<<<blocks, threads, 0, ctx.stream()>>>(
            (const int32_t *) selected->data,
            (const int32_t *) pool_cells->data,
            (const float   *) pool_bias->data,
            (const int32_t *) tail_cells->data,
            (const half    *) tail_mask->data,
            (int32_t       *) dst->data,
            kpool, selected->ne[0], pool_bias->ne[0], tail_cells->ne[0], dst->ne[0], dst->ne[1], n_elements);
    CUDA_CHECK(cudaGetLastError());
}
