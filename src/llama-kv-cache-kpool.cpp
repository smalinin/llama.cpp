#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    return n_kv/kpool + 2*n_seqs;
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_pools, indexer_top_k/kpool);
}

bool llama_kpool_indexed_attn_enabled(int64_t n_kv, int64_t n_tps) {
    if (n_tps <= 1) {
        return false;
    }

    const char * env = std::getenv("LLAMA_GLM5_INDEXED_ATTN");
    const int mode = env ? std::atoi(env) : 1;
    return mode >= 2 || (mode == 1 && (n_tps >= 4096 || n_kv >= 32768));
}

struct llama_kpool_cache::impl {
    struct pool_id_hash {
        size_t operator()(const pool_id & id) const {
            const uint64_t a = (uint32_t) id.seq;
            const uint64_t b = (uint64_t) id.block;
            return (size_t) (a*0x9e3779b97f4a7c15ULL ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
        }
    };

    struct entry {
        int32_t slot;
        bool cached;
    };

    struct layer {
        uint32_t il;
        ggml_tensor * keys;
        ggml_tensor * topk_hint;
        ggml_tensor * mtp_topk;
        ggml_tensor * mtp_mask;
    };

    uint32_t max_slots;
    uint32_t n_stream;
    uint32_t n_embd;
    uint32_t kpool;
    bool rebuild = true;
    bool mtp_index_reuse = false;

    std::vector<std::unordered_map<pool_id, entry, pool_id_hash>> maps;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
};

llama_kpool_cache::llama_kpool_cache(
        const llama_model & model,
        bool                offload,
        bool                unified,
        uint32_t            kv_size,
        uint32_t            n_seq_max,
        uint32_t            kpool,
        uint32_t            n_embd,
        const layer_filter_cb & filter) : pimpl(new impl) {
    GGML_ASSERT(kpool > 1);
    GGML_ASSERT(n_seq_max > 0);
    GGML_ASSERT(n_embd > 0);

    pimpl->n_stream = unified ? 1 : n_seq_max;
    const uint32_t n_ps_max = unified ? n_seq_max : 1;
    pimpl->max_slots = llama_kpool_n_pools(kv_size, kpool, n_ps_max);
    pimpl->n_embd = n_embd;
    pimpl->kpool = kpool;
    pimpl->maps.resize(pimpl->n_stream);

    struct buft_comparator {
        bool operator()(ggml_backend_buffer_type_t lhs, ggml_backend_buffer_type_t rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second.get();
        }

        ggml_init_params params = {
            /*.mem_size   =*/ size_t(4u*model.hparams.n_layer_all*ggml_tensor_overhead()),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            return nullptr;
        }
        ctx_map.emplace(buft, ctx);
        return ctx;
    };

    for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for GLM pool-key cache");
        }

        ggml_tensor * keys = ggml_new_tensor_3d(
                ctx, GGML_TYPE_F32, n_embd, pimpl->max_slots, pimpl->n_stream);
        ggml_format_name(keys, "cache_kpool_l%d", il);

        ggml_tensor * topk_hint = ggml_new_tensor_2d(
                ctx, GGML_TYPE_I32, model.hparams.indexer_top_k/kpool, pimpl->n_stream);
        ggml_format_name(topk_hint, "cache_kpool_topk_l%d", il);

        ggml_tensor * mtp_topk = nullptr;
        ggml_tensor * mtp_mask = nullptr;
        if (il >= model.hparams.n_layer()) {
            mtp_topk = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, model.hparams.indexer_top_k, pimpl->n_stream);
            mtp_mask = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_F16, model.hparams.indexer_top_k, pimpl->n_stream);
            ggml_format_name(mtp_topk, "cache_mtp_topk_l%d", il);
            ggml_format_name(mtp_mask, "cache_mtp_topk_mask_l%d", il);
        }

        pimpl->map_layer_ids[il] = pimpl->layers.size();
        pimpl->layers.push_back({ il, keys, topk_hint, mtp_topk, mtp_mask });
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error("failed to allocate GLM pool-key cache");
        }

        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s GLM pool-key cache buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        pimpl->ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: slots = %u, width = %u, streams = %u, layers = %zu\n",
            __func__, pimpl->max_slots, n_embd, pimpl->n_stream, pimpl->layers.size());
}

llama_kpool_cache::~llama_kpool_cache() = default;

void llama_kpool_cache::clear(bool data) {
    invalidate();
    if (data) {
        for (auto & [_, buf] : pimpl->ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

void llama_kpool_cache::invalidate() {
    for (auto & map : pimpl->maps) {
        map.clear();
    }
    pimpl->rebuild = true;
    pimpl->mtp_index_reuse = false;
}

void llama_kpool_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p0 >= 0 && p1 >= 0 && p0 >= p1) {
        return;
    }

    const int64_t block0 = p0 < 0 ? 0 : p0/pimpl->kpool;
    const int64_t block1 = p1 < 0 ? INT64_MAX : (p1 - 1)/pimpl->kpool;

    for (auto & map : pimpl->maps) {
        for (auto it = map.begin(); it != map.end();) {
            const pool_id & id = it->first;
            if ((seq_id < 0 || id.seq == seq_id) && id.block >= block0 && id.block <= block1) {
                it = map.erase(it);
            } else {
                ++it;
            }
        }
    }
}

bool llama_kpool_cache::needs_rebuild() const {
    return pimpl->rebuild;
}

void llama_kpool_cache::finish_rebuild() {
    pimpl->rebuild = false;
}

void llama_kpool_cache::set_mtp_index_reuse(bool reuse) {
    // Keep the diagnostic A/B switch coherent across graph construction and
    // host input preparation.  If graph reuse is disabled, the normal pool
    // cache update inputs must remain active as well.
    if (reuse) {
        const char * env = std::getenv("LLAMA_GLM5_MTP_TOPK_SHARE");
        reuse = env == nullptr || std::atoi(env) != 0;
    }

    pimpl->mtp_index_reuse = reuse;
}

bool llama_kpool_cache::get_mtp_index_reuse() const {
    return pimpl->mtp_index_reuse;
}

void llama_kpool_cache::mark_dirty(uint32_t stream, llama_seq_id seq_id, llama_pos pos) {
    GGML_ASSERT(stream < pimpl->n_stream);
    GGML_ASSERT(seq_id >= 0 && pos >= 0);

    auto & map = pimpl->maps[stream];
    const auto it = map.find({ seq_id, pos/pimpl->kpool });
    if (it != map.end()) {
        it->second.cached = false;
    }
}

llama_kpool_cache::stream_plan llama_kpool_cache::prepare_stream(
        uint32_t stream,
        const std::vector<pool_id> & ids,
        size_t n_scratch,
        bool rebuild) {
    GGML_ASSERT(stream < pimpl->n_stream);
    GGML_ASSERT(ids.size() + n_scratch <= pimpl->max_slots);

    auto & map = pimpl->maps[stream];
    if (rebuild) {
        map.clear();
    }

    std::unordered_set<pool_id, impl::pool_id_hash> current(ids.begin(), ids.end());
    GGML_ASSERT(current.size() == ids.size() && "pool identities within a stream must be unique");

    for (auto it = map.begin(); it != map.end();) {
        if (current.find(it->first) == current.end()) {
            it = map.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<bool> used(pimpl->max_slots);
    for (const auto & [_, value] : map) {
        GGML_ASSERT(value.slot >= 0 && (uint32_t) value.slot < pimpl->max_slots);
        GGML_ASSERT(!used[value.slot]);
        used[value.slot] = true;
    }

    stream_plan result;
    result.slots.reserve(ids.size());
    result.cached.reserve(ids.size());

    uint32_t next_free = 0;
    auto take_free = [&]() {
        while (next_free < pimpl->max_slots && used[next_free]) {
            ++next_free;
        }
        GGML_ASSERT(next_free < pimpl->max_slots);
        used[next_free] = true;
        return (int32_t) next_free++;
    };

    for (const pool_id & id : ids) {
        auto it = map.find(id);
        if (it == map.end()) {
            it = map.emplace(id, impl::entry { take_free(), false }).first;
        }
        result.slots.push_back(it->second.slot);
        result.cached.push_back(it->second.cached ? 1 : 0);
    }

    result.scratch.reserve(n_scratch);
    for (size_t i = 0; i < n_scratch; ++i) {
        result.scratch.push_back(take_free());
    }

    return result;
}

void llama_kpool_cache::mark_cached(uint32_t stream, const std::vector<int32_t> & slots) {
    GGML_ASSERT(stream < pimpl->n_stream);
    auto & map = pimpl->maps[stream];
    for (int32_t slot : slots) {
        bool found = false;
        for (auto & [_, value] : map) {
            if (value.slot == slot) {
                value.cached = true;
                found = true;
                break;
            }
        }
        GGML_ASSERT(found);
    }
}

uint32_t llama_kpool_cache::get_max_slots() const {
    return pimpl->max_slots;
}

uint32_t llama_kpool_cache::get_n_stream() const {
    return pimpl->n_stream;
}

uint32_t llama_kpool_cache::get_max_uncached(uint32_t stream0, uint32_t n_stream) const {
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    uint32_t result = 0;
    for (uint32_t s = stream0; s < stream0 + n_stream; ++s) {
        uint32_t current = 0;
        for (const auto & [_, entry] : pimpl->maps[s]) {
            current += !entry.cached;
        }
        result = std::max(result, current);
    }
    return result;
}

ggml_tensor * llama_kpool_cache::get(
        ggml_context * ctx,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    const int32_t ic = pimpl->map_layer_ids.at(il);
    ggml_tensor * keys = pimpl->layers[ic].keys;
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return ggml_view_3d(ctx, keys, pimpl->n_embd, pimpl->max_slots, n_stream,
            keys->nb[1], keys->nb[2], stream0*keys->nb[2]);
}

ggml_tensor * llama_kpool_cache::store(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * idxs,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    ggml_tensor * dst = get(ctx, il, stream0, n_stream);
    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(cur->ne[0] == dst->ne[0] && cur->ne[2] == dst->ne[2]);
    return ggml_set_rows(ctx, dst, cur, idxs);
}

ggml_tensor * llama_kpool_cache::get_mtp_selection(
        ggml_context * ctx,
        int32_t il,
        int64_t n_selected,
        uint32_t stream0,
        uint32_t n_stream,
        bool mask) const {
    const int32_t ic = pimpl->map_layer_ids.at(il);
    ggml_tensor * src = mask ? pimpl->layers[ic].mtp_mask : pimpl->layers[ic].mtp_topk;

    GGML_ASSERT(src != nullptr && "MTP selection storage exists only for NextN layers");
    GGML_ASSERT(n_selected > 0 && n_selected <= src->ne[0]);
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return ggml_view_3d(ctx, src, n_selected, 1, n_stream,
            src->nb[1], src->nb[1], stream0*src->nb[1]);
}

ggml_tensor * llama_kpool_cache::store_mtp_selection(
        ggml_context * ctx,
        ggml_tensor * cur,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream,
        bool mask) const {
    GGML_ASSERT(cur->ne[1] == 1 && cur->ne[2] == n_stream &&
            "MTP Top-K sharing is defined for one query per active stream");

    ggml_tensor * dst = get_mtp_selection(ctx, il, cur->ne[0], stream0, n_stream, mask);
    GGML_ASSERT(cur->type == dst->type);
    return ggml_cpy(ctx, cur, dst);
}

ggml_tensor * llama_kpool_cache::get_topk_hint(
        ggml_context * ctx,
        int32_t il,
        int64_t n_pools,
        uint32_t stream0,
        uint32_t n_stream) const {
    const int32_t ic = pimpl->map_layer_ids.at(il);
    ggml_tensor * src = pimpl->layers[ic].topk_hint;

    GGML_ASSERT(n_pools > 0 && n_pools <= src->ne[0]);
    GGML_ASSERT(stream0 + n_stream <= pimpl->n_stream);

    return ggml_view_3d(ctx, src, n_pools, 1, n_stream, src->nb[1], src->nb[1], stream0*src->nb[1]);
}

ggml_tensor * llama_kpool_cache::store_topk_hint(
        ggml_context * ctx,
        ggml_tensor * cur,
        int32_t il,
        uint32_t stream0,
        uint32_t n_stream) const {
    GGML_ASSERT(cur->ne[1] == 1 && cur->ne[2] == n_stream);

    ggml_tensor * dst = get_topk_hint(ctx, il, cur->ne[0], stream0, n_stream);
    GGML_ASSERT(cur->type == dst->type);
    return ggml_cpy(ctx, cur, dst);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kpool_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & [_, buf] : pimpl->ctxs_bufs) {
        result[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return result;
}

// sel_mask and cand_mask hold only 0.0f and -INFINITY, so f16 is exact here
template <typename T> struct kpool_mask_of;

template <> struct kpool_mask_of<float> {
    static float from(float v) { return v; }
};

template <> struct kpool_mask_of<ggml_fp16_t> {
    static ggml_fp16_t from(float v) { return ggml_fp32_to_fp16(v); }
};

template <typename T>
static void kpool_mask_fill(T * dst, int64_t n) {
    std::fill(dst, dst + n, kpool_mask_of<T>::from(-INFINITY));
}

template <typename T>
static void kpool_mask_row(
                T * cur_sel,
                T * cur_cand,
          int32_t * cur_tail_cells,
                T * cur_tail_mask,
        const llama_pos * pos_at,
        const int32_t   * pool_of,
          int64_t   n_kv,
          int64_t   n_tail_pad,
        llama_pos   q,
        llama_pos   tail_start,
          int64_t   bo_vis) {
    const T v_sel  = kpool_mask_of<T>::from(0.0f);
    const T v_mask = kpool_mask_of<T>::from(-INFINITY);

    for (int64_t j = 0; j < n_kv; ++j) {
        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;
        const bool tail   = pos_at[j] >= tail_start;

        if (cur_sel != nullptr) {
            cur_sel[j] = vis && tail ? v_sel : v_mask;
        }
        // the candidate set, which the top-k budget may overrun but must never escape
        if (cur_cand != nullptr) {
            cur_cand[j] = vis && (pooled || tail) ? v_sel : v_mask;
        }

        if (vis && tail && cur_tail_cells != nullptr) {
            const int64_t ti = pos_at[j] - tail_start;
            GGML_ASSERT(ti >= 0 && ti < n_tail_pad);
            cur_tail_cells[ti] = (int32_t) j;
            cur_tail_mask [ti] = v_sel;
        }
    }
}

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
              ggml_tensor    * compact_tail_cells,
              ggml_tensor    * compact_tail_mask,
              llama_kpool_cache * pool_cache,
              ggml_tensor    * pool_cache_slots,
              ggml_tensor    * pool_store_src,
              ggml_tensor    * pool_store_dst,
              ggml_tensor    * pool_update_cells,
              ggml_tensor    * pool_update_dst,
              bool             rebuild_pool_cache,
              uint32_t         stream0) {
    GGML_ASSERT(kv != nullptr);
    GGML_ASSERT(kpool > 0);

    GGML_ASSERT(pool_cells->buffer && "pool_cells input was not allocated");
    GGML_ASSERT(pool_bias ->buffer && "pool_bias input was not allocated");
    GGML_ASSERT((sel_mask == nullptr) == (cand_mask == nullptr));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_bias ->buffer));
    if (sel_mask != nullptr) {
        GGML_ASSERT(sel_mask->buffer && cand_mask->buffer && "dense sparse masks were not allocated");
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask->buffer));
    }

    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias ->type == GGML_TYPE_F32);
    if (sel_mask != nullptr) {
        GGML_ASSERT((sel_mask->type == GGML_TYPE_F16 || sel_mask->type == GGML_TYPE_F32) &&
                "sel_mask must be f16 or f32");
        GGML_ASSERT(cand_mask->type == sel_mask->type && "both masks must have the KQ mask's type");
    }
    GGML_ASSERT((compact_tail_cells == nullptr) == (compact_tail_mask == nullptr));
    GGML_ASSERT((pool_cache == nullptr) == (pool_cache_slots == nullptr));
    if (pool_cache != nullptr) {
        GGML_ASSERT(rebuild_pool_cache ?
                (pool_store_src != nullptr && pool_store_dst != nullptr &&
                 pool_update_cells == nullptr && pool_update_dst == nullptr) :
                (pool_store_src == nullptr && pool_store_dst == nullptr &&
                 pool_update_cells != nullptr && pool_update_dst != nullptr));
    }

    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(pool_bias));
    GGML_ASSERT(sel_mask == nullptr || ggml_is_contiguous(sel_mask));
    GGML_ASSERT(cand_mask == nullptr || ggml_is_contiguous(cand_mask));

    if (compact_tail_cells != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(compact_tail_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(compact_tail_mask ->buffer));
        GGML_ASSERT(compact_tail_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(compact_tail_mask->type == GGML_TYPE_F16 || compact_tail_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask == nullptr || compact_tail_mask->type == sel_mask->type);
        GGML_ASSERT(ggml_is_contiguous(compact_tail_cells));
        GGML_ASSERT(ggml_is_contiguous(compact_tail_mask));
    }

    if (pool_cache != nullptr) {
        GGML_ASSERT(pool_cache_slots->buffer && "pool_cache_slots input was not allocated");
        GGML_ASSERT(ggml_backend_buffer_is_host(pool_cache_slots->buffer));
        GGML_ASSERT(pool_cache_slots->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(pool_cache_slots));

        if (rebuild_pool_cache) {
            GGML_ASSERT(pool_store_src->buffer && "pool_store_src input was not allocated");
            GGML_ASSERT(pool_store_dst->buffer && "pool_store_dst input was not allocated");
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_store_src->buffer));
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_store_dst->buffer));
            GGML_ASSERT(pool_store_src->type == GGML_TYPE_I32 && pool_store_dst->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(pool_store_src) && ggml_is_contiguous(pool_store_dst));
        } else {
            GGML_ASSERT(pool_update_cells->buffer && "pool_update_cells input was not allocated");
            GGML_ASSERT(pool_update_dst->buffer && "pool_update_dst input was not allocated");
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_update_cells->buffer));
            GGML_ASSERT(ggml_backend_buffer_is_host(pool_update_dst->buffer));
            GGML_ASSERT(pool_update_cells->type == GGML_TYPE_I32 && pool_update_dst->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(pool_update_cells) && ggml_is_contiguous(pool_update_dst));
        }
    }

    const int64_t n_ns     = pool_cells->ne[1];
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION] positions are unambiguous only within one sequence, so
    // one pool map per SEQUENCE, not per stream
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;
    const int64_t n_pools = pool_cells->ne[0]/r;

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(sel_mask == nullptr || sel_mask->ne[2] == 1);
    GGML_ASSERT(sel_mask == nullptr || ggml_are_same_shape(cand_mask, sel_mask));
    GGML_ASSERT(pool_bias->ne[0] == n_pools && pool_bias->ne[2] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps      = n_tokens/n_ns;
    const int64_t n_padq     = sel_mask ? sel_mask->ne[1] : n_tps;
    const int64_t n_tail_pad = compact_tail_cells ? compact_tail_cells->ne[0] : 0;

    GGML_ASSERT(pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);

    if (compact_tail_cells != nullptr) {
        GGML_ASSERT(n_tail_pad >= r - 1);
        GGML_ASSERT(compact_tail_cells->ne[1] == n_tps && compact_tail_cells->ne[2] == n_ns);
        GGML_ASSERT(ggml_are_same_shape(compact_tail_mask, compact_tail_cells));
    }

    if (pool_cache != nullptr) {
        GGML_ASSERT(stream0 + n_ns <= pool_cache->get_n_stream());
        GGML_ASSERT(pool_cache_slots->ne[0] == n_pools && pool_cache_slots->ne[1] == n_ns);
        if (rebuild_pool_cache) {
            GGML_ASSERT(ggml_are_same_shape(pool_store_src, pool_cache_slots));
            GGML_ASSERT(ggml_are_same_shape(pool_store_dst, pool_cache_slots));
        } else {
            GGML_ASSERT(pool_update_cells->ne[0] % r == 0 &&
                    pool_update_cells->ne[1] == n_ns);
            GGML_ASSERT(pool_update_dst->ne[0] == pool_update_cells->ne[0]/r &&
                    pool_update_dst->ne[1] == n_ns);
        }
    }

    if (cell_pool) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
        GGML_ASSERT(cell_pool->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(cell_pool));
        GGML_ASSERT(cell_pool->ne[0] == n_kv && cell_pool->ne[1] == n_ns);

        // one row per stream, so a shared cell has nowhere to put its second pool
        GGML_ASSERT(n_ps == 1 && "the per-cell pool view needs one sequence per stream");
    }

    if (bias) {
        GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[1] == n_tps && bias->ne[2] == n_ns);
    }

    int32_t * dst_cell_pool  = cell_pool ? (int32_t *) cell_pool->data : nullptr;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = bias ? (float *) bias->data : nullptr;
    float   * dst_pool_bias  = (float   *) pool_bias ->data;
    char    * dst_sel_mask   = sel_mask  ? (char *) sel_mask ->data : nullptr;
    char    * dst_cand_mask  = cand_mask ? (char *) cand_mask->data : nullptr;

    int32_t * dst_cache_slots  = pool_cache_slots  ? (int32_t *) pool_cache_slots ->data : nullptr;
    int32_t * dst_store_src    = pool_store_src    ? (int32_t *) pool_store_src   ->data : nullptr;
    int32_t * dst_store_dst    = pool_store_dst    ? (int32_t *) pool_store_dst   ->data : nullptr;
    int32_t * dst_update_cells = pool_update_cells ? (int32_t *) pool_update_cells->data : nullptr;
    int32_t * dst_update_dst   = pool_update_dst   ? (int32_t *) pool_update_dst  ->data : nullptr;

    const ggml_type mask_type = sel_mask ? sel_mask->type : compact_tail_mask->type;
    const bool   mask_f16 = mask_type == GGML_TYPE_F16;
    const size_t mask_ts  = ggml_type_size(mask_type);

    int32_t * dst_tail_cells = compact_tail_cells ? (int32_t *) compact_tail_cells->data : nullptr;
    char    * dst_tail_mask  = compact_tail_mask  ? (char    *) compact_tail_mask ->data : nullptr;

    if (compact_tail_cells != nullptr) {
        std::fill(dst_tail_cells, dst_tail_cells + ggml_nelements(compact_tail_cells), 0);
        if (mask_f16) {
            kpool_mask_fill((ggml_fp16_t *) dst_tail_mask, ggml_nelements(compact_tail_mask));
        } else {
            kpool_mask_fill((float *) dst_tail_mask, ggml_nelements(compact_tail_mask));
        }
    }

    // -1 marks a cell with no usable pool; host side only, never copied into cell_pool
    std::vector<int32_t>   pool_of(n_kv);
    std::vector<int32_t>   filled(n_pools);
    std::vector<llama_pos> pos_at;

    std::vector<int64_t> run_off(n_ps);
    std::vector<int64_t> run_len(n_ps);

    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pools);
        char    * cur_sel_mask   = dst_sel_mask  ? dst_sel_mask  + s*(n_padq*n_kv)*mask_ts : nullptr;
        char    * cur_cand_mask  = dst_cand_mask ? dst_cand_mask + s*(n_padq*n_kv)*mask_ts : nullptr;
        float   * cur_pool_bias  = dst_pool_bias  + s*(n_tps*n_pools);

        struct valid_pool {
            llama_kpool_cache::pool_id id;
            int32_t packed;
            std::vector<int32_t> members;
        };
        std::vector<valid_pool> valid_pools;

        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);
        std::fill(cur_pool_bias,  cur_pool_bias  + n_tps*n_pools, -INFINITY);

        // the token loop writes rows < n_tps in full; only the padding rows need clearing
        if (cur_sel_mask != nullptr && mask_f16) {
            kpool_mask_fill((ggml_fp16_t *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            kpool_mask_fill((ggml_fp16_t *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
        } else if (cur_sel_mask != nullptr) {
            kpool_mask_fill((float *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            kpool_mask_fill((float *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
        }

        // [TAG_KPOOL_PACK] one packed run per sequence, sized on the pool range it holds.
        // NOT one full-width table per sequence: the indexer scores every slot against
        // every query, so that multiplies the score tensor by n_seq_max.
        // llama_memory_seq_cp can ask for more slots than exist; then a sequence keeps its
        // newest pools, the same cut a large hole already forces.
        {
            int64_t n_want = 0;

            for (int64_t ps = 0; ps < n_ps; ++ps) {
                const llama_seq_id seq = seq_of(s, ps);
                const auto & cells = kv->get_cells(seq);

                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                        continue;
                    }
                    const int64_t b = cells.pos_get(j)/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                run_len[ps] = found ? b_max - b_min + 1 : 0;
                n_want += run_len[ps];
            }

            if (n_want > n_pools) {
                int64_t rem = n_pools;

                for (int64_t ps = 0; ps < n_ps; ++ps) {
                    run_len[ps] = std::min(run_len[ps], rem/(n_ps - ps));
                    rem -= run_len[ps];
                }
            }

            int64_t off = 0;
            for (int64_t ps = 0; ps < n_ps; ++ps) {
                run_off[ps] = off;
                off += run_len[ps];
            }

            GGML_ASSERT(off <= n_pools);
        }

        int64_t n_done = 0;

        for (int64_t ps = 0; ps < n_ps; ++ps) {
            const llama_seq_id seq_of_pool = seq_of(s, ps);
            const auto & cells = kv->get_cells(seq_of_pool);

            const int64_t n_run = run_len[ps];

            int32_t * cur_cell_pool   = dst_cell_pool ? dst_cell_pool + s*n_kv : nullptr;
            int32_t * part_pool_cells = cur_pool_cells + run_off[ps]*r;

            std::fill(pool_of.begin(), pool_of.end(), -1);
            std::fill(filled.begin(),  filled.end(),   0);

            pos_at.resize(n_kv);
            for (int64_t j = 0; j < n_kv; ++j) {
                pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_pool) ? -1 : cells.pos_get(j);
            }

            // anchoring at the absolute p/kpool follows vLLM and SGLang, not HF
            // (valid_keys.argmax(-1)): it is the only anchor that keeps a pool's identity
            // stable from the prefill that built it to the decodes that read it.
            int64_t b_base = 0;
            {
                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (pos_at[j] < 0) {
                        continue;
                    }
                    const int64_t b = pos_at[j]/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                b_base = std::max(b_min, b_max - (n_run - 1));
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }

                const llama_pos p  = pos_at[j];
                const int64_t   bo = p/r - b_base;

                if (bo < 0 || bo >= n_run) {
                    continue;
                }

                pool_of[j] = (int32_t) bo;
                part_pool_cells[bo*r + (p%r)] = (int32_t) j;
                filled[bo]++;
            }

            // pool_valid = grouped_valid_keys.all(-1): the compressor consumes all r keys
            for (int64_t j = 0; j < n_kv; ++j) {
                // != rather than <: two cells claiming one position overwrite each other
                if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                    pool_of[j] = -1;
                }
                if (cur_cell_pool) {
                    cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
                }
            }

            if (pool_cache != nullptr) {
                for (int64_t p = 0; p < n_run; ++p) {
                    if (filled[p] != (int32_t) r) {
                        continue;
                    }

                    valid_pool item;
                    item.id = { seq_of_pool, b_base + p };
                    item.packed = (int32_t) (run_off[ps] + p);
                    item.members.assign(part_pool_cells + p*r, part_pool_cells + (p + 1)*r);
                    valid_pools.push_back(std::move(item));
                }
            }

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int64_t   i = s*n_tps + ii;

                if (ubatch->seq_id[i][0] != seq_of_pool) {
                    continue;
                }

                const llama_pos q = ubatch->pos[i];

                // q >= 0 is what makes the unsigned range test below a range test
                GGML_ASSERT(q >= 0);

                n_done++;

                // index_kpool_always_select_tail, which lands selection on pool boundaries
                const llama_pos tail_start = (q + 1)/r*r;

                // the reference tests visibility at a pool's LAST member, so a pool the
                // query straddles is dropped whole
                const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;
                char  * cur_sel  = cur_sel_mask  ? cur_sel_mask  + ii*n_kv*mask_ts : nullptr;
                char  * cur_cand = cur_cand_mask ? cur_cand_mask + ii*n_kv*mask_ts : nullptr;
                int32_t * cur_tail_cells = dst_tail_cells ?
                        dst_tail_cells + (s*n_tps + ii)*n_tail_pad : nullptr;
                char * cur_tail_mask = dst_tail_mask ?
                        dst_tail_mask + (s*n_tps + ii)*n_tail_pad*mask_ts : nullptr;

                if (mask_f16) {
                    kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                            cur_tail_cells, (ggml_fp16_t *) cur_tail_mask,
                            pos_at.data(), pool_of.data(), n_kv, n_tail_pad, q, tail_start, bo_vis);
                } else {
                    kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                            cur_tail_cells, (float *) cur_tail_mask,
                            pos_at.data(), pool_of.data(), n_kv, n_tail_pad, q, tail_start, bo_vis);
                }

                if (cur_bias) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;

                        cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                    }
                }

                // the query's own sequence run only; every other slot keeps the -INFINITY
                // of the fill above, which is what keeps a foreign pool out of the budget
                float * q_pool_bias = cur_pool_bias + ii*n_pools + run_off[ps];

                for (int64_t p = 0; p < n_run; ++p) {
                    const bool valid   = filled[p] == (int32_t) r;
                    const bool visible = p < bo_vis;

                    q_pool_bias[p] = valid && visible ? 0.0f : -INFINITY;
                }
            }
        }

        // exactly one partition per row, or a query reads another sequence's pools
        GGML_ASSERT(n_done == n_tps && "every query must belong to a sequence of the ubatch");

        if (pool_cache != nullptr) {
            std::vector<llama_kpool_cache::pool_id> ids;
            ids.reserve(valid_pools.size());
            for (const valid_pool & pool : valid_pools) {
                ids.push_back(pool.id);
            }

            const size_t n_scratch = rebuild_pool_cache ? n_pools - valid_pools.size() : 0;
            auto plan = pool_cache->prepare_stream(stream0 + s, ids, n_scratch, rebuild_pool_cache);

            int32_t * cur_cache_slots = dst_cache_slots + s*n_pools;
            std::fill(cur_cache_slots, cur_cache_slots + n_pools, 0);

            for (size_t i = 0; i < valid_pools.size(); ++i) {
                cur_cache_slots[valid_pools[i].packed] = plan.slots[i];
            }

            std::vector<int32_t> cached_now;

            if (rebuild_pool_cache) {
                int32_t * cur_store_src = dst_store_src + s*n_pools;
                int32_t * cur_store_dst = dst_store_dst + s*n_pools;

                size_t i = 0;
                for (; i < valid_pools.size(); ++i) {
                    cur_store_src[i] = valid_pools[i].packed;
                    cur_store_dst[i] = plan.slots[i];
                    cached_now.push_back(plan.slots[i]);
                }
                for (size_t j = 0; i < (size_t) n_pools; ++i, ++j) {
                    cur_store_src[i] = 0;
                    cur_store_dst[i] = plan.scratch[j];
                }
            } else {
                const size_t n_update = pool_update_cells->ne[0]/r;
                std::vector<size_t> uncached;
                for (size_t i = 0; i < valid_pools.size(); ++i) {
                    if (!plan.cached[i]) {
                        uncached.push_back(i);
                    }
                }
                GGML_ASSERT(uncached.size() <= n_update &&
                        "incremental pool-key update capacity is too small");

                // Inputs are laid out [r, n_update, n_stream]. Pad unused
                // update rows by recomputing an existing valid pool; writing
                // the same value to the same slot is side-effect free.
                const size_t fallback = valid_pools.empty() ? SIZE_MAX : 0;
                int32_t * cur_update_cells = dst_update_cells + s*r*n_update;
                int32_t * cur_update_dst   = dst_update_dst   + s*n_update;

                for (size_t u = 0; u < n_update; ++u) {
                    const size_t chosen = u < uncached.size() ? uncached[u] : fallback;
                    int32_t * row = cur_update_cells + u*r;
                    if (chosen != SIZE_MAX) {
                        std::copy(valid_pools[chosen].members.begin(), valid_pools[chosen].members.end(), row);
                        cur_update_dst[u] = plan.slots[chosen];
                    } else {
                        std::fill(row, row + r, 0);
                        cur_update_dst[u] = 0;
                    }
                }

                for (size_t chosen : uncached) {
                    cached_now.push_back(plan.slots[chosen]);
                }
            }

            pool_cache->mark_cached(stream0 + s, cached_now);
        }
    }

    if (pool_cache != nullptr && rebuild_pool_cache) {
        pool_cache->finish_rebuild();
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    // unconditional: the key and gate STORE runs on the dense path too. gating it the
    // way the scoring is gated would leave every cell below n_select with no indexer
    // state, and the first ubatch to cross n_select would pool cells never written
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    if (pool_cells == nullptr) {
        return;
    }

    // The compact suffix is consumed by indexed prefill and decode. On a dense
    // compatibility graph it is pruned and intentionally has no allocator buffer.
    ggml_tensor * tail_cells = compact_tail_cells && compact_tail_cells->buffer ? compact_tail_cells : nullptr;
    ggml_tensor * tail_mask  = compact_tail_mask  && compact_tail_mask ->buffer ? compact_tail_mask  : nullptr;
    GGML_ASSERT((tail_cells == nullptr) == (tail_mask == nullptr));

    // Reuse steps need a fresh compact tail map, but deliberately do not
    // mutate completed pool keys. Mark only pools touched by speculative
    // positions dirty so catch-up recompresses them without rebuilding the
    // full long-context cache.
    if (pool_cache != nullptr && pool_cache->get_mtp_index_reuse()) {
        const int64_t n_stream = pool_cells->ne[1];
        GGML_ASSERT(n_stream > 0 && ubatch->n_tokens%n_stream == 0);
        const int64_t n_tps = ubatch->n_tokens/n_stream;

        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t i = s*n_tps; i < (s + 1)*n_tps; ++i) {
                GGML_ASSERT(ubatch->n_seq_id[i] > 0);
                pool_cache->mark_dirty(stream0 + s, ubatch->seq_id[i][0], ubatch->pos[i]);
            }
        }
    }

    llama_kpool_cache * active_pool_cache =
            pool_cache != nullptr && !pool_cache->get_mtp_index_reuse() ? pool_cache : nullptr;

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias,
            sel_mask, cand_mask, ubatch, kpool, mctx_attn->get_n_kv(), tail_cells, tail_mask,
            active_pool_cache,
            active_pool_cache ? pool_cache_slots  : nullptr,
            active_pool_cache ? pool_store_src    : nullptr,
            active_pool_cache ? pool_store_dst    : nullptr,
            active_pool_cache ? pool_update_cells : nullptr,
            active_pool_cache ? pool_update_dst   : nullptr,
            active_pool_cache ? rebuild_pool_cache : false,
            mctx_idx->get_stream_base());
}

bool llm_graph_input_kpool::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);
    const auto * attn = mctx->get_attn();
    const auto * idx  = mctx->get_idx();
    auto * cache = mctx->get_kpool_cache();

    const char * pool_cache_env = std::getenv("LLAMA_GLM5_POOL_CACHE");
    if (pool_cache_env != nullptr && std::atoi(pool_cache_env) == 0) {
        cache = nullptr;
    }

    bool res = true;
    res &= attn == mctx_attn;
    res &= idx == mctx_idx;
    res &= cache == pool_cache;
    res &= idx->get_stream_base() == stream0;
    res &= k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= params.hparams.indexer_kpool == kpool;

    if (pool_cells == nullptr) {
        return res;
    }

    const int64_t n_kv = attn->get_n_kv();
    const int64_t n_stream = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
    if (n_stream <= 0 || params.ubatch.n_tokens%n_stream != 0 || params.ubatch.n_seqs_unq%n_stream != 0) {
        return false;
    }

    const int64_t n_tps = params.ubatch.n_tokens/n_stream;
    const int64_t n_ps = params.ubatch.n_seqs_unq/n_stream;
    const int64_t n_pools = llama_kpool_n_pools(n_kv, kpool, n_ps);
    const int64_t n_pool_select = llama_kpool_select_k(
            n_pools, params.hparams.indexer_top_k, params.hparams.indexer_kpool);
    const int64_t n_selected = kpool*n_pool_select;
    const int64_t n_compact = GGML_PAD(n_selected + kpool - 1, 256);
    const bool indexed_attn = llama_kpool_indexed_attn_enabled(n_kv, n_tps);
    const bool compact_decode = n_tps == 1 && n_kv >= n_compact;
    const bool dense_masks = !indexed_attn && !compact_decode;

    auto shape = [](const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return t != nullptr && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == ne3;
    };

    res &= shape(pool_cells, kpool*n_pools, n_stream, 1, 1);
    res &= shape(pool_bias, n_pools, n_tps, n_stream, 1);
    res &= shape(compact_tail_cells, n_compact - n_selected, n_tps, n_stream, 1);
    res &= shape(compact_tail_mask, n_compact - n_selected, n_tps, n_stream, 1);

    if (pool_bias_f16 != nullptr) {
        res &= shape(pool_bias_f16, n_pools, n_tps, 1, n_stream);
    }

    res &= dense_masks == (sel_mask != nullptr && cand_mask != nullptr);
    if (dense_masks) {
        res &= shape(sel_mask, n_kv, n_tps, 1, n_stream);
        res &= shape(cand_mask, n_kv, n_tps, 1, n_stream);
    }

    const bool rebuild = cache != nullptr && cache->needs_rebuild();
    const bool mtp_reuse = cache != nullptr && cache->get_mtp_index_reuse();
    res &= rebuild == rebuild_pool_cache;
    res &= mtp_reuse == mtp_index_reuse;

    if (cache != nullptr) {
        res &= shape(pool_cache_slots, n_pools, n_stream, 1, 1);
        if (rebuild) {
            res &= shape(pool_store_src, n_pools, n_stream, 1, 1);
            res &= shape(pool_store_dst, n_pools, n_stream, 1, 1);
            res &= pool_update_cells == nullptr && pool_update_dst == nullptr;
        } else {
            const int64_t n_pending = cache->get_max_uncached(stream0, n_stream);
            const int64_t n_update = std::min<int64_t>(
                    n_pools, (n_tps + kpool - 1)/kpool + n_ps + n_pending);
            res &= shape(pool_update_cells, kpool*n_update, n_stream, 1, 1);
            res &= shape(pool_update_dst, n_update, n_stream, 1, 1);
            res &= pool_store_src == nullptr && pool_store_dst == nullptr;
        }
    }

    return res;
}
