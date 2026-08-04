#include "common.cuh"

void ggml_cuda_op_msa_block_mask(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *        idx,
        const ggml_tensor *        mask,
        ggml_tensor *              dst);
