#pragma once

#include "common.cuh"

void ggml_cuda_kpool_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_kpool_expand_supported(const ggml_tensor * dst);
