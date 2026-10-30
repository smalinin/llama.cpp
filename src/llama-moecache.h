#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, dummy_slot+1] live next to the router or a resident projection;
//    the last slice stays zero; a writable spare is enabled unless LLAMA_MOE_CACHE_SPARE=0.
//  - an I32 table maps expert id -> slot, or dummy_slot when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - fully host-resident layers split the whole FFN between CPU and GPU.
//    Mixed layers merge each cached projection before the nonlinearity.
//    CPU and GPU quantized matmuls can have different rounding.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).
// LLAMA_MOE_CACHE_PROFILE=1 prints per-device counters when the cache is freed.

#include "ggml-backend.h"

#include <cstdint>
#include <map>

// Larger batches use the stock prefill path.
constexpr int32_t LLAMA_MOE_CACHE_MAX_TOKENS = 32;
constexpr int32_t LLAMA_MOE_STAGE_DEFAULT_TOKENS = 512;
constexpr int32_t LLAMA_MOE_STAGE_MAX_TOKENS = 8192;
constexpr int32_t LLAMA_MOE_STAGE_EXPERTS = 4;

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0; // Useful resident capacity; excludes spare and dummy.
    int32_t dummy_slot = 0; // Fixed last slice; writable slots are below it.
    int32_t n_batch = 1; // Device batch limit for repeated dummy IDs.

    // Original weights; mixed layers can have device-resident projections.
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    const ggml_tensor * observe_src = nullptr;

    // Null for a projection already on device; otherwise ne[2] == dummy_slot + 1.
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or dummy_slot when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;

    // Shared per-device workspace, enabled by LLAMA_MOE_PREFILL_STAGE=1.
    ggml_tensor * stage_weights = nullptr;
    ggml_tensor * stage_input = nullptr;
    ggml_tensor * stage_output = nullptr;
    ggml_tensor * stage_ids = nullptr;
    ggml_tensor * stage_host = nullptr;
    int32_t stage_min_tokens = 33;
    int32_t stage_max_tokens = 0; // Frozen with the workspace allocation.
    bool stage_auto = false;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t threads = 0);

// Fixed per-layer partitions under one CUDA device budget. Returns the maximum layer capacity.
// Includes alignment, dummy/spare slots, device tables and same-device staging. Host and remote
// staging allocations remain in memory_breakdown but do not consume this device budget.
struct llama_moe_profile_snapshot;
int32_t llama_moe_cache_init_budget(const llama_model & model, uint64_t bytes,
        ggml_backend_dev_t device, int32_t max_inserts, int32_t threads = 0,
        const char * profile = nullptr, const llama_moe_profile_snapshot * snapshot = nullptr);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// apply throttled LRU updates; call between graph executions only
void llama_moe_cache_step(const llama_model & model, ggml_backend_sched_t sched);

void llama_moe_cache_free(const llama_model & model);

// Metadata probes never open profile files. Collection reuses existing host IDs.
struct llama_moe_profile_snapshot;
void llama_moe_cache_profile_init(const llama_model & model, const char * path,
        const llama_moe_profile_snapshot * snapshot = nullptr);
void llama_moe_cache_profile_begin(const llama_model & model, int64_t tokens);
void llama_moe_cache_preload(const llama_model & model);

// Actual cache buffers, or expected sizes for a no_alloc model.
std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache_memory_breakdown(const llama_model & model, int32_t n_slots);
