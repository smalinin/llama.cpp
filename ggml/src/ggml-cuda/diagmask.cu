#include "diagmask.cuh"

static __global__ void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.y*blockIdx.y + threadIdx.y;
    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void diag_mask_inf_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(nrows_x, block_num_x, 1);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int nrows0 = ggml_nrows(src0);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_cuda(src0_d, dst_d, ne00, nrows0, ne01, n_past, stream);
}


template <typename T>
static __global__ void prefix_mask(const int32_t * visible, T * dst, int ncols, int n) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = T(i%ncols < visible[i/ncols] ? 0.0f : -INFINITY);
    }
}

bool ggml_cuda_prefix_mask_supported(const ggml_tensor * op) {
    return op->src[0] && op->src[0]->type == GGML_TYPE_I32 &&
        ggml_is_vector(op->src[0]) && ggml_is_contiguous(op->src[0]) && op->src[0]->nb[0] == sizeof(int32_t) &&
        (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
        ggml_is_matrix(op) && ggml_is_contiguous(op) && op->nb[0] == ggml_type_size(op->type) &&
        op->ne[0] > 0 && op->ne[1] > 0 &&
        op->ne[1] == op->src[0]->ne[0] && ggml_nelements(op) <= INT32_MAX;
}

void ggml_cuda_op_prefix_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_prefix_mask_supported(dst));
    const auto * visible = (const int32_t *) dst->src[0]->data;
    const int64_t n = ggml_nelements(dst);
    const unsigned blocks = (n + 255)/256;
    if (dst->type == GGML_TYPE_F16) {
        prefix_mask<<<blocks, 256, 0, ctx.stream()>>>(visible, (half *) dst->data, dst->ne[0], n);
    } else {
        prefix_mask<<<blocks, 256, 0, ctx.stream()>>>(visible, (float *) dst->data, dst->ne[0], n);
    }
}
