#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

enum class llm_sparse_selection_unit : uint8_t {
    token,
    compressed_entry,
    pool,
    block,
};

enum class llm_sparse_score_transform : uint8_t {
    identity,
    relu,
};

enum class llm_sparse_score_reduction : uint8_t {
    none,
    head_sum,
    weighted_head_sum,
};

enum class llm_sparse_causal_policy : uint8_t {
    none,
    mask,
    score_bias,
};

enum class llm_sparse_tail_policy : uint8_t {
    none,
    dense_incomplete_group,
    local_window,
};

enum class llm_sparse_expansion_policy : uint8_t {
    none,
    mask,
    gather,
    group_members,
};

enum class llm_sparse_cache_location : uint8_t {
    none,
    device,
    host,
    split,
};

enum llm_sparse_transport : uint32_t {
    LLM_SPARSE_TRANSPORT_NONE         = 0,
    LLM_SPARSE_TRANSPORT_SAME_DEVICE  = 1u << 0,
    LLM_SPARSE_TRANSPORT_P2P          = 1u << 1,
    LLM_SPARSE_TRANSPORT_HOST_STAGING = 1u << 2,
};

enum class llm_sparse_selection_mode : uint8_t {
    dense,
    sparse_mask,
    sparse_gather,
    reuse,
};

struct llm_sparse_selection_desc {
    const char * name = nullptr;

    llm_sparse_selection_unit unit = llm_sparse_selection_unit::token;

    uint32_t q_head_count = 0;
    uint32_t k_head_count = 0;
    uint32_t head_dim     = 0;

    llm_sparse_score_transform score_transform = llm_sparse_score_transform::identity;
    llm_sparse_score_reduction score_reduction = llm_sparse_score_reduction::none;
    float score_scale = 1.0f;

    uint64_t dense_threshold = 0;
    uint64_t requested_top_k = 0;
    uint32_t group_size      = 1;

    llm_sparse_causal_policy causal_policy       = llm_sparse_causal_policy::none;
    llm_sparse_tail_policy tail_policy           = llm_sparse_tail_policy::none;
    llm_sparse_expansion_policy expansion_policy = llm_sparse_expansion_policy::none;

    int32_t owner_layer  = -1;
    int32_t source_layer = -1;
    bool reuse_allowed   = false;

    llm_sparse_cache_location cache_location = llm_sparse_cache_location::none;
    uint32_t allowed_transports = LLM_SPARSE_TRANSPORT_NONE;
};

struct llm_sparse_selection_request {
    uint64_t available_units = 0;
    uint32_t dense_tail_rows = 0;
    bool prefer_gather       = false;
    bool reuse_available     = false;
    bool allow_dense         = false;
};

struct llm_sparse_selection_decision {
    llm_sparse_selection_mode mode = llm_sparse_selection_mode::dense;
    uint64_t effective_top_k = 0;
    uint64_t selected_rows   = 0;
    uint32_t dense_tail_rows = 0;
};

bool llm_sparse_selection_validate(
        const llm_sparse_selection_desc & desc,
        std::string * error = nullptr);

llm_sparse_selection_decision llm_sparse_selection_dispatch(
        const llm_sparse_selection_desc & desc,
        const llm_sparse_selection_request & request);

bool llm_sparse_selection_cuda_lid_capable(
        const llm_sparse_selection_desc & desc,
        ggml_type cache_type);

void llm_sparse_selection_profile(
        const llm_sparse_selection_desc & desc,
        const llm_sparse_selection_request & request,
        const llm_sparse_selection_decision & decision);

const char * llm_sparse_selection_mode_name(llm_sparse_selection_mode mode);
