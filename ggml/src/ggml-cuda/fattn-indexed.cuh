#pragma once

#include "ggml-cuda.h"

void ggml_cuda_flash_attn_ext_indexed(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_indexed_supported(const ggml_tensor * dst);
