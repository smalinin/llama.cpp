#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_concat_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * shared, ggml_tensor * dst);
