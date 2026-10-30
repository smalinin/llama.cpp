#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
struct llama_model;
struct llama_layer;
// Calibrate at startup, persist costs beside the GGUF, and return nullptr when CPU wins.
ggml_backend_buffer_type_t llama_moe_stage_auto(const llama_model & model,
        ggml_backend_buffer_type_t home, const std::vector<const llama_layer *> & layers,
        int32_t threads, int32_t max_tokens, size_t cache_bytes,
        std::map<ggml_backend_dev_t, size_t> & reserved);
