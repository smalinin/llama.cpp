#include "llama-sparse-selection.h"

#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {

template<typename T>
bool fail(std::string * error, const T & message) {
    if (error) {
        *error = message;
    }
    return false;
}

const char * unit_name(llm_sparse_selection_unit unit) {
    switch (unit) {
        case llm_sparse_selection_unit::token:            return "token";
        case llm_sparse_selection_unit::compressed_entry: return "compressed_entry";
        case llm_sparse_selection_unit::pool:             return "pool";
        case llm_sparse_selection_unit::block:            return "block";
    }
    return "unknown";
}

const char * transform_name(llm_sparse_score_transform transform) {
    switch (transform) {
        case llm_sparse_score_transform::identity: return "identity";
        case llm_sparse_score_transform::relu:     return "relu";
    }
    return "unknown";
}

const char * reduction_name(llm_sparse_score_reduction reduction) {
    switch (reduction) {
        case llm_sparse_score_reduction::none:              return "none";
        case llm_sparse_score_reduction::head_sum:          return "head_sum";
        case llm_sparse_score_reduction::weighted_head_sum: return "weighted_head_sum";
    }
    return "unknown";
}

const char * causal_name(llm_sparse_causal_policy policy) {
    switch (policy) {
        case llm_sparse_causal_policy::none:       return "none";
        case llm_sparse_causal_policy::mask:       return "mask";
        case llm_sparse_causal_policy::score_bias: return "score_bias";
    }
    return "unknown";
}

const char * tail_name(llm_sparse_tail_policy policy) {
    switch (policy) {
        case llm_sparse_tail_policy::none:                   return "none";
        case llm_sparse_tail_policy::dense_incomplete_group: return "dense_incomplete_group";
        case llm_sparse_tail_policy::local_window:           return "local_window";
    }
    return "unknown";
}

const char * expansion_name(llm_sparse_expansion_policy policy) {
    switch (policy) {
        case llm_sparse_expansion_policy::none:          return "none";
        case llm_sparse_expansion_policy::mask:          return "mask";
        case llm_sparse_expansion_policy::gather:        return "gather";
        case llm_sparse_expansion_policy::group_members: return "group_members";
    }
    return "unknown";
}

const char * cache_name(llm_sparse_cache_location location) {
    switch (location) {
        case llm_sparse_cache_location::none:   return "none";
        case llm_sparse_cache_location::device: return "device";
        case llm_sparse_cache_location::host:   return "host";
        case llm_sparse_cache_location::split:  return "split";
    }
    return "unknown";
}

} // namespace

const char * llm_sparse_selection_mode_name(llm_sparse_selection_mode mode) {
    switch (mode) {
        case llm_sparse_selection_mode::dense:         return "dense";
        case llm_sparse_selection_mode::sparse_mask:   return "sparse_mask";
        case llm_sparse_selection_mode::sparse_gather: return "sparse_gather";
        case llm_sparse_selection_mode::reuse:         return "reuse";
    }
    return "unknown";
}

bool llm_sparse_selection_validate(const llm_sparse_selection_desc & desc, std::string * error) {
    if (desc.name == nullptr || desc.name[0] == '\0') {
        return fail(error, "selection name is empty");
    }
    if (desc.q_head_count == 0 || desc.k_head_count == 0 || desc.head_dim == 0) {
        return fail(error, "Q/K head shape must be non-zero");
    }
    if (!std::isfinite(desc.score_scale) || desc.score_scale <= 0.0f) {
        return fail(error, "score scale must be finite and positive");
    }
    if (desc.requested_top_k == 0) {
        return fail(error, "requested top-k must be non-zero");
    }
    if (desc.group_size == 0) {
        return fail(error, "selection group size must be non-zero");
    }
    if ((desc.unit == llm_sparse_selection_unit::pool || desc.unit == llm_sparse_selection_unit::block) &&
            desc.group_size == 1) {
        return fail(error, "pool/block selection requires a group size greater than one");
    }
    if (desc.tail_policy == llm_sparse_tail_policy::dense_incomplete_group && desc.group_size == 1) {
        return fail(error, "an incomplete-group tail requires grouped selection");
    }
    if (desc.expansion_policy == llm_sparse_expansion_policy::group_members && desc.group_size == 1) {
        return fail(error, "group-member expansion requires grouped selection");
    }
    if (desc.owner_layer < 0 || desc.source_layer < 0) {
        return fail(error, "owner and source layers must be specified");
    }
    if (desc.reuse_allowed && desc.source_layer > desc.owner_layer) {
        return fail(error, "a reusable selection cannot read from a future layer");
    }
    if (desc.cache_location == llm_sparse_cache_location::none) {
        if (desc.allowed_transports != LLM_SPARSE_TRANSPORT_NONE) {
            return fail(error, "a selection without a cache cannot declare transport paths");
        }
    } else if (desc.allowed_transports == LLM_SPARSE_TRANSPORT_NONE) {
        return fail(error, "a cached selection must declare at least one transport path");
    }
    return true;
}

llm_sparse_selection_decision llm_sparse_selection_dispatch(
        const llm_sparse_selection_desc & desc,
        const llm_sparse_selection_request & request) {
    std::string error;
    if (!llm_sparse_selection_validate(desc, &error)) {
        throw std::runtime_error(std::string(desc.name ? desc.name : "sparse selection") + ": " + error);
    }
    if (request.available_units == 0) {
        throw std::runtime_error(std::string(desc.name) + ": no selectable units are available");
    }
    if (desc.tail_policy != llm_sparse_tail_policy::dense_incomplete_group && request.dense_tail_rows != 0) {
        throw std::runtime_error(std::string(desc.name) + ": dense tail rows do not match the tail policy");
    }
    if (request.dense_tail_rows >= desc.group_size && request.dense_tail_rows != 0) {
        throw std::runtime_error(std::string(desc.name) + ": dense tail must be smaller than one group");
    }
    if (request.reuse_available && !desc.reuse_allowed) {
        throw std::runtime_error(std::string(desc.name) + ": reuse requested by a non-reusable selection");
    }

    llm_sparse_selection_decision result;
    result.effective_top_k = std::min(desc.requested_top_k, request.available_units);
    result.dense_tail_rows = request.dense_tail_rows;

    if (result.effective_top_k >
            (std::numeric_limits<uint64_t>::max() - result.dense_tail_rows)/desc.group_size) {
        throw std::runtime_error(std::string(desc.name) + ": expanded selection width overflows");
    }
    result.selected_rows = result.effective_top_k*desc.group_size + result.dense_tail_rows;

    if (request.reuse_available) {
        result.mode = llm_sparse_selection_mode::reuse;
    } else if (request.allow_dense &&
            (request.available_units <= desc.dense_threshold ||
             result.effective_top_k == request.available_units)) {
        result.mode = llm_sparse_selection_mode::dense;
    } else if (request.prefer_gather) {
        result.mode = llm_sparse_selection_mode::sparse_gather;
    } else {
        result.mode = llm_sparse_selection_mode::sparse_mask;
    }

    return result;
}

bool llm_sparse_selection_cuda_lid_capable(
        const llm_sparse_selection_desc & desc,
        ggml_type cache_type) {
    if (desc.head_dim != 128 || (desc.q_head_count != 32 && desc.q_head_count != 64) ||
            desc.k_head_count != 1 || desc.score_transform != llm_sparse_score_transform::relu ||
            desc.score_reduction != llm_sparse_score_reduction::weighted_head_sum) {
        return false;
    }

    switch (cache_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

void llm_sparse_selection_profile(
        const llm_sparse_selection_desc & desc,
        const llm_sparse_selection_request & request,
        const llm_sparse_selection_decision & decision) {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_SPARSE_PROFILE");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (!enabled) {
        return;
    }

    LLAMA_LOG_INFO(
            "selection_contract: name=%s unit=%s q_heads=%u k_heads=%u head_dim=%u"
            " transform=%s reduction=%s scale=%.9g dense_threshold=%" PRIu64
            " requested_top_k=%" PRIu64 " available_units=%" PRIu64
            " effective_top_k=%" PRIu64 " group_size=%u selected_rows=%" PRIu64
            " dense_tail_rows=%u causal=%s tail=%s expansion=%s mode=%s"
            " owner_layer=%d source_layer=%d reuse_allowed=%d reuse_available=%d"
            " cache=%s transports=0x%x\n",
            desc.name, unit_name(desc.unit), desc.q_head_count, desc.k_head_count, desc.head_dim,
            transform_name(desc.score_transform), reduction_name(desc.score_reduction), desc.score_scale,
            desc.dense_threshold, desc.requested_top_k, request.available_units,
            decision.effective_top_k, desc.group_size, decision.selected_rows,
            decision.dense_tail_rows, causal_name(desc.causal_policy), tail_name(desc.tail_policy),
            expansion_name(desc.expansion_policy), llm_sparse_selection_mode_name(decision.mode),
            desc.owner_layer, desc.source_layer, desc.reuse_allowed, request.reuse_available,
            cache_name(desc.cache_location), desc.allowed_transports);
}
