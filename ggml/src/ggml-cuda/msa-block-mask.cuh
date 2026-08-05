#include "common.cuh"

void ggml_cuda_op_msa_block_top_k(
        ggml_backend_cuda_context & ctx,
        ggml_tensor *              dst);

bool ggml_cuda_msa_block_top_k_supported(
        int                 device,
        const ggml_tensor * dst);

void ggml_cuda_op_msa_sparse_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor *              dst);

bool ggml_cuda_msa_sparse_attn_supported(const ggml_tensor * dst);

void ggml_cuda_op_msa_block_mask(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *        idx,
        const ggml_tensor *        mask,
        ggml_tensor *              dst);

void ggml_cuda_op_msa_block_mask_mapped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *        idx,
        const ggml_tensor *        cell_block,
        const ggml_tensor *        mask,
        ggml_tensor *              dst);
