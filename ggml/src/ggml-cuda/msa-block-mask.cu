#include "msa-block-mask.cuh"

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
        const int * __restrict__ idx,
        const int * __restrict__ cell_block,
        const char * __restrict__ mask,
        half * __restrict__ dst,
        const int n_kv,
        const int n_tokens,
        const int n_heads,
        const int k,
        const size_t mask_nb1) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;

    const int * idx_row = idx + k * (head + n_heads * token);
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
    GGML_ASSERT(ggml_is_contiguous(idx));
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
            (const int *) idx->data, (const int *) cell_block->data,
            (const char *) mask->data, (half *) dst->data,
            n_kv, n_tokens, n_heads, k, mask->nb[1]);
}
