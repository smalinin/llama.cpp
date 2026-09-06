#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;
class llama_model;

// Persistent F32 storage for completed GLM-5-Next pool keys. Pool identities are
// tracked host-side, while the key tensors live beside their model layers.
class llama_kpool_cache {
public:
    struct pool_id {
        llama_seq_id seq;
        int64_t      block;

        bool operator==(const pool_id & other) const {
            return seq == other.seq && block == other.block;
        }
    };

    struct stream_plan {
        std::vector<int32_t> slots;
        std::vector<uint8_t> cached;
        std::vector<int32_t> scratch;
    };

    using layer_filter_cb = std::function<bool(uint32_t)>;

    llama_kpool_cache(
            const llama_model & model,
            bool                offload,
            bool                unified,
            uint32_t            kv_size,
            uint32_t            n_seq_max,
            uint32_t            kpool,
            uint32_t            n_embd,
            const layer_filter_cb & filter);

    ~llama_kpool_cache();

    void clear(bool data);
    void invalidate();
    void seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    bool needs_rebuild(uint32_t stream0, uint32_t n_stream) const;
    void finish_rebuild(uint32_t stream0, uint32_t n_stream);

    void set_mtp_index_reuse(bool reuse);
    bool get_mtp_index_reuse() const;
    void mark_dirty(uint32_t stream, llama_seq_id seq_id, llama_pos pos);

    stream_plan prepare_stream(
            uint32_t stream,
            const std::vector<pool_id> & ids,
            size_t n_scratch,
            bool rebuild);

    void mark_cached(uint32_t stream, const std::vector<int32_t> & slots);

    uint32_t get_max_slots() const;
    uint32_t get_n_stream() const;
    uint32_t get_max_uncached(uint32_t stream0, uint32_t n_stream) const;

    ggml_tensor * get(
            ggml_context * ctx,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

    ggml_tensor * store(
            ggml_context * ctx,
            ggml_tensor * cur,
            ggml_tensor * idxs,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

    ggml_tensor * get_mtp_selection(
            ggml_context * ctx,
            int32_t il,
            int64_t n_selected,
            uint32_t stream0,
            uint32_t n_stream,
            bool mask) const;

    ggml_tensor * store_mtp_selection(
            ggml_context * ctx,
            ggml_tensor * cur,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream,
            bool mask) const;

    ggml_tensor * get_topk_hint(
            ggml_context * ctx,
            int32_t il,
            int64_t n_pools,
            uint32_t stream0,
            uint32_t n_stream) const;

    ggml_tensor * store_topk_hint(
            ggml_context * ctx,
            ggml_tensor * cur,
            int32_t il,
            uint32_t stream0,
            uint32_t n_stream) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// GLM-5-Next indexer pooling. the position -> cell map is built host side because
// find_slot's cell order is arbitrary. no input may hold a negative index: ggml_set_rows
// asserts i1 >= 0 and ggml_get_rows has no sentinel, so unusable entries are clamped into
// range and neutralised by the additive masks instead.

// pool slots for `n_kv` cells shared by `n_seqs` sequences: n_kv/kpool, exact only while
// the sequences' cells are disjoint, plus 2 per sequence for rebasing.
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of modular_glm5_next.py, Glm5NextTextIndexer.forward. must run over POOLS, not
// cells: relu ties span pool boundaries, so a cell-level cut takes partial pools.
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// Mode 0 uses dense attention, mode 1 selects by KV and batch size, and mode 2 forces indexed attention.
bool llama_kpool_indexed_attn_enabled(int64_t n_kv, int64_t n_tps);

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
//   cell_pool  I32 [n_kv, n_stream]                 per-cell view, optional, unused here
//   pool_cells I32 [kpool*n_pools, n_stream]        pool member -> cell, 0 if not resident
//   bias       F32 [n_kv, n_tps, n_stream]          per-cell view, optional, unused here
//   pool_bias  F32 [n_pools, n_tps, n_stream]       pool_valid & pool_visible, -INFINITY
//       outside the query's own sequence run; computed, not gathered from `bias` at the
//       last member, which an incomplete pool lacks and would inherit cell 0's validity
//   sel_mask   F16/F32 [n_kv, n_batch, 1, n_stream] 0.0f on the always-selected tail only
//   cand_mask  F16/F32 [n_kv, n_batch, 1, n_stream] max(bias, sel_mask); bounds the top-k
//       spills that a partial seq_rm would otherwise let escape the candidate set
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool,
              uint32_t         n_kv,
              ggml_tensor    * compact_tail_cells = nullptr,
              ggml_tensor    * compact_tail_mask  = nullptr,
              llama_kpool_cache * pool_cache      = nullptr,
              ggml_tensor    * pool_cache_slots   = nullptr,
              ggml_tensor    * pool_store_src     = nullptr,
              ggml_tensor    * pool_store_dst     = nullptr,
              ggml_tensor    * pool_update_cells  = nullptr,
              ggml_tensor    * pool_update_dst    = nullptr,
              bool             rebuild_pool_cache = false,
              uint32_t         stream0            = 0);

// One pooling map per ubatch; rebuilding it per indexer layer costs O(n_kv * n_tokens)
// host writes and dominates prefill. sharing is valid only while every indexer layer sees
// the same candidate set - true for glm5next (indexer_types all "full"), not for windowed.
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            llama_kpool_cache * pool_cache,
            bool rebuild_pool_cache,
            bool mtp_index_reuse,
            uint32_t stream0,
            uint32_t kpool) :
        mctx_attn(mctx_attn),
        mctx_idx(mctx_idx),
        pool_cache(pool_cache),
        rebuild_pool_cache(rebuild_pool_cache),
        mtp_index_reuse(mtp_index_reuse),
        stream0(stream0),
        kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    // The direct compact-attention suffix: the real incomplete tail followed by
    // masked cell-zero padding up to the CUDA FA stride. Shared by every layer.
    ggml_tensor * compact_tail_cells = nullptr; // I32 [n_tail_pad, n_tps, n_stream]
    ggml_tensor * compact_tail_mask  = nullptr; // F16 [n_tail_pad, n_tps, n_stream]

    ggml_tensor * pool_cache_slots  = nullptr; // I32 [n_pools, n_stream]
    ggml_tensor * pool_store_src    = nullptr; // I32 [n_pools, n_stream], rebuild only
    ggml_tensor * pool_store_dst    = nullptr; // I32 [n_pools, n_stream], rebuild only
    ggml_tensor * pool_update_cells = nullptr; // I32 [kpool*n_update, n_stream], incremental
    ggml_tensor * pool_update_dst   = nullptr; // I32 [n_update, n_stream], incremental

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;
    llama_kpool_cache * pool_cache;

    const bool rebuild_pool_cache;
    const bool mtp_index_reuse;
    const uint32_t stream0;
    const uint32_t kpool;
};
