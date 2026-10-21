#pragma once
#include "common.cuh"

bool ggml_cuda_dsv41_attn_indices_supported(int device, const ggml_tensor * dst);
void ggml_cuda_dsv41_attn_indices(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
