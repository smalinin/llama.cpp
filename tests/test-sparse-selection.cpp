#include "llama-sparse-selection.h"

#include <cstdlib>
#include <iostream>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

static llm_sparse_selection_desc make_lid_desc(uint32_t n_heads = 64) {
    llm_sparse_selection_desc desc;
    desc.name                = "test_lid";
    desc.unit                = llm_sparse_selection_unit::compressed_entry;
    desc.q_head_count        = n_heads;
    desc.k_head_count        = 1;
    desc.head_dim            = 128;
    desc.score_transform     = llm_sparse_score_transform::relu;
    desc.score_reduction     = llm_sparse_score_reduction::weighted_head_sum;
    desc.score_scale         = 1.0f/128.0f;
    desc.requested_top_k     = 512;
    desc.group_size          = 1;
    desc.causal_policy       = llm_sparse_causal_policy::mask;
    desc.expansion_policy    = llm_sparse_expansion_policy::mask;
    desc.owner_layer         = 12;
    desc.source_layer        = 8;
    desc.reuse_allowed       = true;
    desc.cache_location      = llm_sparse_cache_location::device;
    desc.allowed_transports  = LLM_SPARSE_TRANSPORT_SAME_DEVICE | LLM_SPARSE_TRANSPORT_P2P;
    return desc;
}

int main() {
    std::string error;
    auto lid = make_lid_desc();
    require(llm_sparse_selection_validate(lid, &error), "valid LID descriptor was rejected");

    llm_sparse_selection_request request;
    request.available_units = 300;
    auto decision = llm_sparse_selection_dispatch(lid, request);
    require(decision.mode == llm_sparse_selection_mode::sparse_mask, "LID mask dispatch mismatch");
    require(decision.effective_top_k == 300, "effective top-k was not clamped");
    require(decision.selected_rows == 300, "ungrouped selection width mismatch");

    request.reuse_available = true;
    decision = llm_sparse_selection_dispatch(lid, request);
    require(decision.mode == llm_sparse_selection_mode::reuse, "reusable selection did not dispatch to reuse");

    auto pool = make_lid_desc();
    pool.name               = "test_pool";
    pool.unit               = llm_sparse_selection_unit::pool;
    pool.requested_top_k    = 128;
    pool.group_size         = 4;
    pool.tail_policy        = llm_sparse_tail_policy::dense_incomplete_group;
    pool.expansion_policy   = llm_sparse_expansion_policy::group_members;
    pool.reuse_allowed      = false;
    pool.source_layer       = pool.owner_layer;

    request = {};
    request.available_units = 96;
    request.dense_tail_rows = 3;
    request.prefer_gather   = true;
    decision = llm_sparse_selection_dispatch(pool, request);
    require(decision.mode == llm_sparse_selection_mode::sparse_gather, "pool gather dispatch mismatch");
    require(decision.effective_top_k == 96, "pool top-k clamp mismatch");
    require(decision.selected_rows == 387, "pool expansion width mismatch");

    auto block = pool;
    block.name             = "test_block";
    block.unit             = llm_sparse_selection_unit::block;
    block.dense_threshold  = 128;
    request                = {};
    request.available_units = 64;
    request.allow_dense     = true;
    decision = llm_sparse_selection_dispatch(block, request);
    require(decision.mode == llm_sparse_selection_mode::dense, "dense threshold dispatch mismatch");

    for (uint32_t n_heads : {32u, 64u}) {
        auto capability = make_lid_desc(n_heads);
        for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16}) {
            require(llm_sparse_selection_cuda_lid_capable(capability, type),
                    "supported CUDA LID shape/type was rejected");
        }
    }
    auto unsupported = make_lid_desc(16);
    require(!llm_sparse_selection_cuda_lid_capable(unsupported, GGML_TYPE_F16),
            "unsupported CUDA LID head count was accepted");
    unsupported = make_lid_desc();
    unsupported.score_reduction = llm_sparse_score_reduction::head_sum;
    require(!llm_sparse_selection_cuda_lid_capable(unsupported, GGML_TYPE_F16),
            "QSA reduction was incorrectly accepted as Lightning Indexer");

    auto invalid = pool;
    invalid.group_size = 1;
    require(!llm_sparse_selection_validate(invalid, &error), "invalid grouped descriptor was accepted");
    invalid = lid;
    invalid.allowed_transports = LLM_SPARSE_TRANSPORT_NONE;
    require(!llm_sparse_selection_validate(invalid, &error), "cached descriptor without transport was accepted");
    invalid = lid;
    invalid.source_layer = invalid.owner_layer + 1;
    require(!llm_sparse_selection_validate(invalid, &error), "future reusable source was accepted");

    std::cout << "sparse selection contract: PASS\n";
    return 0;
}
