#include "llama-hparams.h"
#include "llama-impl.h"
#include "models.h"
#include "deepseek41-ops.h"

#include "llama-kv-cache-dsv4.h"
#include "llama-mmap.h"
#include "llama-sparse-selection.h"

#include <algorithm>
#include <climits>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// DeepSeek-V4.1.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache. Three things differ.
//
// 1. The hyper-connection coefficients lag by one sublayer. A sublayer computes the mix that the
//    NEXT one consumes, so attention uses what the previous layer's FFN produced. V4 computes and
//    consumes in the same sublayer.
// 2. There is no learned hyper-connection head. Nothing is left to collapse the copies with in V4,
//    which is why it needs one; here the last layer's FFN mix is still unused and does the job.
//    These two facts explain each other, and the file ships no output_hc_* tensors.
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// The sparse attention also differs. V4 compresses KV on every layer at one of two fixed ratios;
// V4.1 compresses on a few source layers and the layers after each source read the same rows, and
// it derives index keys from that shared latent rather than from a second compressor. See
// build_attention_v41().
//
// The indexer uses the reference's two levels: a source layer first keeps the best compressed
// blocks, and later index-source layers choose individual rows only from those candidates.

static bool dsv41_default_on(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr || (std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0);
}

static constexpr int64_t dsv41_compact_min_compressed = 24576;

static bool dsv41_compact_decode(int64_t n_tokens, int64_t n_stream, int64_t n_comp, int64_t n_top_k, int64_t n_raw, int64_t n_swa) {
    return n_tokens == 1 && n_stream == 1 && n_comp >= dsv41_compact_min_compressed && n_comp < (1LL << 24) &&
        dsv41_can_select_kv(n_comp, n_top_k, n_raw, n_swa) &&
        dsv41_default_on("LLAMA_DSV41_COMPACT_ATTN");
}

static bool dsv41_opt_in(const char * name) {
    const char * value = std::getenv(name);
    return value && (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0);
}

// mean over the hyper-connection copies; deepseek4.cpp keeps its own copy of this
static ggml_tensor * dsv41_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

// Experimental A/B harness. Keep the GGUF value as the production default.
static uint32_t dsv41_indexer_top_k_override() {
    const char * value = std::getenv("LLAMA_DSV41_INDEX_TOP_K");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    uint64_t parsed = 0;
    for (const char * p = value; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            throw std::runtime_error("LLAMA_DSV41_INDEX_TOP_K must be 512, 768, 1024, or 2048");
        }
        parsed = parsed*10 + (uint64_t) (*p - '0');
        if (parsed > UINT32_MAX) {
            throw std::runtime_error("LLAMA_DSV41_INDEX_TOP_K must be 512, 768, 1024, or 2048");
        }
    }

    if (parsed != 512 && parsed != 768 && parsed != 1024 && parsed != 2048) {
        throw std::runtime_error("LLAMA_DSV41_INDEX_TOP_K must be 512, 768, 1024, or 2048");
    }
    return (uint32_t) parsed;
}

int llama_model_deepseek41::engram_index(int il) const {
    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] == (uint32_t) il) {
            return (int) e;
        }
    }
    return -1;
}

static std::vector<float> dsv41_weight_f32(const ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    std::vector<float> values(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    } else {
        const auto * traits = ggml_get_type_traits(tensor->type);
        if (!traits->to_float) {
            throw std::runtime_error(format("cannot convert Engram gate weight %s from %s to F32",
                    tensor->name, ggml_type_name(tensor->type)));
        }
        std::vector<uint8_t> data(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
        traits->to_float(data.data(), values.data(), values.size());
    }
    return values;
}

bool llama_model_deepseek41::load_tensors(llama_model_loader & ml) {
    if (!llama_model_base::load_tensors(ml)) {
        return false;
    }

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> contexts;
    engram_gate_weights.resize(hparams.n_layer(), nullptr);
    engram_gate_ones.resize(hparams.n_layer(), nullptr);
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto & layer = layers[il];
        if (!layer.engram_q && !layer.engram_gate) {
            continue;
        }
        const auto buft = ggml_backend_dev_buffer_type(dev_layer(il));
        auto & ctx = contexts[buft];
        if (!ctx) {
            ctx.reset(ggml_init({2*engram_n_layer*ggml_tensor_overhead(), nullptr, true}));
            if (!ctx) {
                throw std::runtime_error("cannot create Engram gate weight context");
            }
        }
        engram_gate_weights[il] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hparams.n_embd, hparams.dsv4_hc_mult);
        engram_gate_ones[il] = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_format_name(engram_gate_weights[il], "blk.%u.engram_qk_f32", il);
        ggml_format_name(engram_gate_ones[il], "blk.%u.engram_one", il);
    }
    for (auto & [buft, ctx] : contexts) {
        ggml_backend_buffer_ptr buffer(ml.no_alloc ? ggml_backend_buft_alloc_buffer(buft, 0) :
                ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer) {
            throw std::runtime_error("cannot allocate Engram gate weights");
        }
        if (ml.no_alloc) {
            for (auto * t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buffer.get();
            }
        }
        add_tensor_context(std::move(ctx), std::move(buffer));
    }
    if (!ml.no_alloc) {
        const float one = 1.0f;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!engram_gate_weights[il]) {
                continue;
            }
            const auto & layer = layers[il];
            auto q = dsv41_weight_f32(layer.engram_gate ? layer.engram_gate : layer.engram_q);
            if (!layer.engram_gate) {
                const auto k = dsv41_weight_f32(layer.engram_k);
                GGML_ASSERT(q.size() == k.size());
                for (size_t i = 0; i < q.size(); ++i) {
                    q[i] *= k[i];
                }
            }
            ggml_backend_tensor_set(engram_gate_weights[il], q.data(), 0, q.size()*sizeof(float));
            ggml_backend_tensor_set(engram_gate_ones[il], &one, 0, sizeof(one));
        }
    }
    return true;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    auto get_candidate_key = [&](llm_kv key, const char * alias, uint32_t & value) {
        uint32_t other = 0;
        const bool found = ml.get_key(key, value, false);
        const int64_t id = gguf_find_key(ml.metadata, alias);
        bool found_alias;
        if (id >= 0 && gguf_get_kv_type(ml.metadata, id) == GGUF_TYPE_INT32) {
            int32_t signed_value = 0;
            found_alias = ml.get_key(alias, signed_value, false);
            if (signed_value < 0) {
                throw std::runtime_error(format("negative DeepSeek-V4.1 candidate metadata: %s", alias));
            }
            other = (uint32_t) signed_value;
        } else {
            found_alias = ml.get_key(alias, other, false);
        }
        if (found && found_alias && value != other) {
            throw std::runtime_error(format("conflicting DeepSeek-V4.1 candidate metadata: %s", alias));
        }
        if (!found && found_alias) {
            value = other;
        }
        return found || found_alias;
    };
    uint32_t candidate_source_layer = 0;
    const bool has_candidate_source = get_candidate_key(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER_ID,
            "deepseek41.attention.indexer.candidate_source_layer", candidate_source_layer);
    const bool has_candidate_block = get_candidate_key(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,
            "deepseek41.attention.indexer.candidate_block_size", hparams.dsv41_candidate_block_size);
    const bool has_candidate_top_k = get_candidate_key(LLM_KV_ATTENTION_CANDIDATE_TOP_K_BLOCKS,
            "deepseek41.attention.indexer.candidate_top_k_blocks", hparams.dsv41_candidate_top_k_blocks);
    const int candidate_key_count = has_candidate_source + has_candidate_block + has_candidate_top_k;

    if (candidate_key_count != 0 && candidate_key_count != 3) {
        throw std::runtime_error("DeepSeek-V4.1 candidate mask metadata must contain source layer, block size, and top-k blocks");
    }

    if (candidate_key_count == 3) {
        if (candidate_source_layer >= hparams.n_layer()) {
            throw std::runtime_error(format("DeepSeek-V4.1 candidate source layer %u is out of range",
                                            candidate_source_layer));
        }
        if (hparams.dsv41_candidate_block_size == 0 || hparams.dsv41_candidate_top_k_blocks == 0) {
            throw std::runtime_error("DeepSeek-V4.1 candidate block size and top-k blocks must be positive");
        }
        if (hparams.dsv41_candidate_block_size > INT_MAX || hparams.dsv41_candidate_top_k_blocks > INT_MAX) {
            throw std::runtime_error("DeepSeek-V4.1 candidate block size and top-k blocks must fit in int32");
        }
        hparams.dsv41_candidate_source_layer = (int32_t) candidate_source_layer;
    } else if (hparams.n_layer() == 40 && hparams.n_ctx_train >= 1024*1024 && hparams.indexer_top_k == 512) {
        // Compatibility for the public GGUFs converted before the three candidate keys were
        // emitted. These values are checkpoint-specific and verified against its reference config.
        hparams.dsv41_candidate_source_layer = 20;
        hparams.dsv41_candidate_block_size   = 8;
        hparams.dsv41_candidate_top_k_blocks = 2048;
        LLAMA_LOG_WARN("%s: legacy DeepSeek-V4.1 GGUF has no candidate metadata; using checkpoint defaults 20/8/2048\n",
                __func__);
    }

    if (hparams.dsv41_has_candidate_mask()) {
        if ((uint32_t) hparams.dsv41_candidate_source_layer >= hparams.n_layer()) {
            throw std::runtime_error(format("DeepSeek-V4.1 candidate source layer %d is out of range",
                                            hparams.dsv41_candidate_source_layer));
        }
        if (hparams.dsv41_candidate_top_k_blocks > UINT32_MAX/hparams.dsv41_candidate_block_size) {
            throw std::runtime_error("DeepSeek-V4.1 candidate mask threshold overflows uint32");
        }
    }

    if (const uint32_t top_k_override = dsv41_indexer_top_k_override()) {
        if (!hparams.dsv41_has_candidate_mask()) {
            throw std::runtime_error(
                    "LLAMA_DSV41_INDEX_TOP_K requires explicit or checkpoint-compatible candidate mask metadata");
        }
        const uint32_t file_top_k = hparams.indexer_top_k;
        hparams.indexer_top_k = top_k_override;
        LLAMA_LOG_WARN(
                "%s: experimental indexer top-k override %u -> %u; candidate mask remains %d/%u/%u\n",
                __func__, file_top_k, hparams.indexer_top_k,
                hparams.dsv41_candidate_source_layer,
                hparams.dsv41_candidate_block_size,
                hparams.dsv41_candidate_top_k_blocks);
    }

    const bool ple_metadata = !ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, engram_n_layer, false);
    const auto layers_key = ple_metadata ? LLM_KV_PLE_LAYERS : LLM_KV_ENGRAM_LAYER_IDS;
    if (ple_metadata) {
        ml.get_arr_n(layers_key, engram_n_layer);
    }
    if (engram_n_layer == 0 || engram_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("DeepSeek-V4.1 engram layer count %u is out of range", engram_n_layer));
    }
    ml.get_arr(layers_key, hparams.engram_layer_ids);

    ml.get_key(ple_metadata ? LLM_KV_PLE_HEADS_PER_NGRAM : LLM_KV_ENGRAM_HEAD_COUNT, hparams.engram_n_head);
    ml.get_key(ple_metadata ? LLM_KV_EMBEDDING_LENGTH_PER_LAYER : LLM_KV_ENGRAM_KEY_LENGTH, hparams.engram_key_length);
    ml.get_key(ple_metadata ? LLM_KV_PLE_NGRAM_SIZE : LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);
    ml.get_key(ple_metadata ? "deepseek41.engram.pad_token_id" : ml.llm_kv(LLM_KV_ENGRAM_PAD_ID), engram_pad_id);
    engram_shared_table = ple_metadata || ml.get_weight("per_layer_token_embd.weight") ||
            gguf_find_tensor(ml.metadata, "per_layer_token_embd.weight") >= 0;

    if (hparams.engram_n_head == 0 || hparams.engram_key_length == 0 || hparams.engram_max_ngram_size < 2) {
        throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head and a 2-gram");
    }

    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] >= hparams.n_layer()) {
            throw std::runtime_error(format("engram layer %u is out of range", hparams.engram_layer_ids[e]));
        }
        for (uint32_t prev = 0; prev < e; ++prev) {
            if (hparams.engram_layer_ids[prev] == hparams.engram_layer_ids[e]) {
                throw std::runtime_error("duplicate DeepSeek-V4.1 engram layer");
            }
        }
    }

    ml.get_arr(ple_metadata ? LLM_KV_PLE_LAYER_MULTIPLIERS : LLM_KV_ENGRAM_MULTIPLIERS, engram_multipliers);
    ml.get_arr(ple_metadata ? LLM_KV_PLE_HEAD_VOCAB_SIZES : LLM_KV_ENGRAM_PRIMES, engram_primes);
    ml.get_arr(ple_metadata ? LLM_KV_PLE_HEAD_OFFSETS : LLM_KV_ENGRAM_OFFSETS, engram_offsets);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,   engram_token_map);
    if (ple_metadata) {
        if (engram_pad_id >= engram_token_map.size() || engram_token_map[engram_pad_id] < 0) {
            throw std::runtime_error("invalid DeepSeek-V4.1 engram pad token");
        }
        engram_pad_id = (uint32_t) engram_token_map[engram_pad_id];
    }

    // the hash indexes straight into these, so a short array would read past the end
    const size_t n_bucket = (size_t) (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;

    if (engram_multipliers.size() != (size_t) engram_n_layer * hparams.engram_max_ngram_size) {
        throw std::runtime_error("engram multiplier count does not match layers * ngram size");
    }
    if (engram_primes.size() != (size_t) engram_n_layer * n_bucket ||
        engram_offsets.size() != engram_primes.size()) {
        throw std::runtime_error("engram prime or offset count does not match layers * buckets");
    }
    for (size_t i = 0; i < engram_primes.size(); ++i) {
        if (engram_primes[i] == 0 || engram_primes[i] > INT32_MAX ||
                engram_offsets[i] > (uint64_t) INT32_MAX - engram_primes[i]) {
            throw std::runtime_error("engram bucket range must fit positive int32 row indices");
        }
    }
    if (engram_shared_table) {
        LLAMA_LOG_INFO("%s: using shared Engram table and precombined gate weights\n", __func__);
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head   = hparams.n_embd_head_k();
    const int64_t o_groups      = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult       = hparams.dsv4_hc_mult;
    const int64_t hc_dim        = hc_mult * n_embd;
    const int64_t hc_mix_dim    = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_indexer = hparams.indexer_head_size;

    if ((size_t) n_vocab > engram_token_map.size()) {
        throw std::runtime_error(format("engram token map has %zu entries, too few for %" PRId64 " tokens",
                                        engram_token_map.size(), n_vocab));
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    if (engram_shared_table) {
        uint64_t max_row = 0;
        for (size_t b = 0; b < engram_offsets.size(); ++b) {
            max_row = std::max(max_row, engram_offsets[b] + engram_primes[b]);
        }
        const auto name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight");
        const auto * w = ml.get_weight(name.str().c_str());
        const int64_t n_rows = w ? w->tensor->ne[1] : (int64_t) max_row;
        if ((uint64_t) n_rows < max_row) {
            throw std::runtime_error("shared Engram table is too short for its hash buckets");
        }
        per_layer_tok_embd = create_tensor(name, {hparams.engram_key_length, n_rows}, TENSOR_READ_LAZY);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // the file lays wo_a out as (n_head * n_embd_head / o_groups, o_lora_rank * o_groups),
        // so reshape at load and keep the graph free of it
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // Only the KV source layers carry a compressor, and only those with a ratio above 1 pool
        // with a gate, so both are optional rather than keyed off the ratio the way V4 does it.
        layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, TENSOR_NOT_REQUIRED);

        // An index source scores queries against shared index keys. Only a layer that also
        // compresses its own KV builds those keys; the rest read what an earlier layer published.
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd_head, n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);

        const int eg = engram_index(i);
        if (eg >= 0) {
            const int64_t n_cols = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
            const int64_t key_len = hparams.engram_key_length;

            if (engram_shared_table) {
                layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_KV, "weight", i), {n_cols * key_len, n_embd * (hc_mult + 1)}, 0);
                layer.engram_gate = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE, "weight", i), {n_embd, hc_mult}, 0);
                continue;
            }

            // a row index is a bucket offset plus a hash, so the last bucket has to end inside
            uint64_t max_row = 0;
            for (int64_t b = 0; b < n_cols; ++b) {
                const size_t k = (size_t) eg*n_cols + b;
                max_row = std::max(max_row, engram_offsets[k] + engram_primes[k]);
            }

            // The table has hundreds of millions of rows and is far too large to hold in memory,
            // but each token only touches n_cols of them, so read those rows on demand.
            const std::string embd_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i).str();
            const auto *      embd_w    = ml.get_weight(embd_name.c_str());
            // llama_model_init_from_user() has no weight directory to query. In that mode the
            // smallest table described by the hash buckets is sufficient for tensor creation.
            // A file-backed model still fails in create_tensor() below if the tensor is absent.
            const int64_t     n_rows    = embd_w ? embd_w->tensor->ne[1] : (int64_t) max_row;

            if ((int64_t) max_row > n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the engram buckets (%" PRIu64 ")",
                                                embd_name.c_str(), n_rows, max_row));
            }

            layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {key_len, n_rows}, TENSOR_READ_LAZY);
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i), {n_cols * key_len, n_embd * (hc_mult + 1)}, 0);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, 0);
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, 0);
        }

    }

    // Work out which layer publishes the stream each layer reads. Only a source carries a
    // compressor, only an index key owner carries indexer_attn_k, and only an index source
    // carries indexer_attn_q_b, so the file itself says which layer plays which role.
    hparams.dsv41_kv_source.fill(-1);
    hparams.dsv41_index_key_source.fill(-1);
    hparams.dsv41_topk_source.fill(-1);

    int32_t last_kv_source    = -1;
    int32_t last_key_owner    = -1;
    int32_t last_index_source = -1;

    for (int i = 0; i < n_layer; ++i) {
        const auto & layer = layers[i];

        if (layer.attn_comp_wkv)    { last_kv_source    = i; }
        if (layer.indexer_attn_k)   { last_key_owner    = i; }
        if (layer.indexer_attn_q_b) { last_index_source = i; }

        if (hparams.dsv4_compress_ratios[i] == 0) {
            // pure sliding window, no compressed stream to read
            continue;
        }

        if (last_kv_source < 0 || last_key_owner < 0 || last_index_source < 0) {
            throw std::runtime_error(format("layer %d reads a compressed stream before any layer publishes one", i));
        }

        // the row layout of a stream follows the ratio it was compressed at, so a reader that
        // disagrees with its source would index into rows that stand for different positions
        if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[last_kv_source]) {
            throw std::runtime_error(format("layer %d compresses at ratio %u but reads layer %d, compressed at %u",
                                            i, hparams.dsv4_compress_ratios[i],
                                            last_kv_source, hparams.dsv4_compress_ratios[last_kv_source]));
        }
        if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[last_key_owner]) {
            throw std::runtime_error(format("layer %d uses ratio %u but reads index keys from layer %d with ratio %u",
                                            i, hparams.dsv4_compress_ratios[i],
                                            last_key_owner, hparams.dsv4_compress_ratios[last_key_owner]));
        }
        if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[last_index_source]) {
            throw std::runtime_error(format("layer %d uses ratio %u but reads top-k from layer %d with ratio %u",
                                            i, hparams.dsv4_compress_ratios[i],
                                            last_index_source, hparams.dsv4_compress_ratios[last_index_source]));
        }

        hparams.dsv41_kv_source[i]        = last_kv_source;
        hparams.dsv41_index_key_source[i] = last_key_owner;
        hparams.dsv41_topk_source[i]      = last_index_source;
    }

    if (hparams.dsv41_has_candidate_mask() &&
            !hparams.dsv41_is_index_source((uint32_t) hparams.dsv41_candidate_source_layer)) {
        throw std::runtime_error(format("DeepSeek-V4.1 candidate source layer %d has no indexer",
                                        hparams.dsv41_candidate_source_layer));
    }

    // a compressor with no gate only makes sense where there is nothing to pool
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.dsv41_is_kv_source(i) && !layers[i].attn_comp_wgate && hparams.dsv4_compress_ratios[i] != 1) {
            throw std::runtime_error(format("layer %d compresses %u tokens per row but has no pooling gate",
                                            i, hparams.dsv4_compress_ratios[i]));
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence, and the compressed token map folds case and accents together first.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model_deepseek41 & pmodel,
                           const llama_kv_cache_dsv4_raw_context * mctx,
                           int eg) : pmodel(pmodel), mctx(mctx), eg(eg) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (pmodel.hparams.engram_max_ngram_size - 1) * pmodel.hparams.engram_n_head;
        return rows->ne[0] == n_cols * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [n_cols * n_tokens]

    const llama_model_deepseek41 & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is, so the right multipliers and buckets are used
    const int eg;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_head;
    const int64_t n_cols   = (n_gram - 1) * n_heads;
    const int64_t n_prev   = n_gram - 1;

    const uint64_t * mult = pmodel.engram_multipliers.data() + (size_t) eg*n_gram;
    const uint64_t * prime = pmodel.engram_primes.data()  + (size_t) eg*n_cols;
    const uint64_t * offset = pmodel.engram_offsets.data() + (size_t) eg*n_cols;

    // an image arrives as an embd batch, so ubatch->token is null; the reference gives those
    // positions no engram contribution at all, which the padding token stands in for here
    const int32_t pad = (int32_t) pmodel.engram_pad_id;
    auto map_of = [&](llama_token t) -> uint64_t {
        if (t < 0 || (size_t) t >= pmodel.engram_token_map.size()) {
            return (uint64_t) pad;
        }
        return (uint64_t) pmodel.engram_token_map[t];
    };

    std::vector<int32_t> idx(n_cols * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // look-back stops at the start of the sequence; everything from there on reads as padding
        std::vector<uint64_t> ctx(n_gram);
        ctx[0] = ubatch->token ? map_of(ubatch->token[i]) : (uint64_t) pad;
        bool blocked = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = blocked ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx[s] = blocked ? (uint64_t) pad : map_of(t);
        }

        // compressed ids stay under 2^17 and the multipliers under 2^37, so no product overflows
        uint64_t rolling = ctx[0] * mult[0];
        for (int64_t s = 1; s < n_gram; ++s) {
            rolling ^= ctx[s] * mult[s];

            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t b = (s - 1)*n_heads + h;
                idx[i*n_cols + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }

    // Hashing is complete for the whole physical batch before the graph reads
    // a table row. Give the OS all random pages together so it can overlap the
    // storage reads instead of servicing one synchronous major fault per row.
    // Keep decode-sized batches out of this path: future generated tokens are
    // unknown and thousands of advice syscalls would only add overhead there.
    static const int64_t prefetch_min_tokens = []() {
        const char * env = std::getenv("LLAMA_DSV41_ENGRAM_PREFETCH");
        if (env != nullptr && (
                std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 ||
                std::strcmp(env, "false") == 0)) {
            return INT64_MAX;
        }
        if (env != nullptr && (
                std::strcmp(env, "1") == 0 || std::strcmp(env, "on") == 0 ||
                std::strcmp(env, "always") == 0)) {
            return int64_t(1);
        }
        return int64_t(32);
    }();

    if (n_tokens >= prefetch_min_tokens) {
        const int il = (int) hp.engram_layer_ids[eg];
        const ggml_tensor * table = pmodel.engram_table(il);
        const size_t row_size = ggml_row_size(table->type, table->ne[0]);

        std::vector<const void *> addrs;
        std::vector<size_t> sizes;
        addrs.reserve(idx.size());
        sizes.reserve(idx.size());

        const uint8_t * data = (const uint8_t *) table->data;
        for (int32_t row : idx) {
            GGML_ASSERT(row >= 0 && row < table->ne[1]);
            addrs.push_back(data + (size_t) row * table->nb[1]);
            sizes.push_back(row_size);
        }
        llama_prefetch_ranges(addrs.data(), sizes.data(), addrs.size());
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

ggml_tensor * llama_model_deepseek41::graph::build_inp_engram(
        const llama_model & model,
        int il) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_engram>(pmodel, mctx_cur->get_raw(), pmodel.engram_index(il));

    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_set_input(inp->rows);
    ggml_tensor * rows = ggml_view_1d(ctx0, inp->rows, inp->rows->ne[0], 0);
    cb(rows, "engram_rows", il);
    res->add_input(std::move(inp));

    // gather then flatten, laying the buckets out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, pmodel.engram_table(il), rows);
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);
    ggml_tensor * h = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx0, x, GGML_TYPE_F32);
    key = key->type == GGML_TYPE_F32 ? key : ggml_cast(ctx0, key, GGML_TYPE_F32);
    key = ggml_reshape_3d(ctx0, key, n_embd, hc, nt);
    // Gate.forward applies inverse RMS factors after the weighted dot-product reduction.
    ggml_tensor * s = dsv41_engram_dot(ctx0, h, key,
            pmodel.engram_gate_weights.at(il), pmodel.engram_gate_ones.at(il), norm_rms_eps);

    // Match copysign, including the distinct signs of +0.0 and -0.0.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_signbit(ctx0, s), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, sign, mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

// Rope settings for one layer. A layer that reads a compressed stream rotates with the
// compressor's base and YaRN; a plain sliding window layer rotates with the model's.
struct dsv41_rope_cfg {
    float   base;
    float   scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t n_ctx_orig;
};

dsv41_rope_cfg llama_model_deepseek41::graph::rope_cfg(int il) const {
    if (hparams.dsv4_compress_ratios[il] == 0) {
        return { freq_base, 1.0f, 0.0f, dsv4_rope_attn_factor(1.0f, 0.0f), 0.0f, 0.0f, 0 };
    }

    return {
        hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
        dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow, n_ctx_orig,
    };
}

// Undo the rotation the query carried into attention, then the grouped output projection. wo_a is
// block diagonal over groups, each projecting only its own heads, hence a batched mul_mat.
ggml_tensor * llama_model_deepseek41::graph::build_attention_tail(
        const llama_model & model,
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = (n_head/n_groups)*n_embd_head;

    const dsv41_rope_cfg rc = rope_cfg(il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);

    ggml_tensor * oa = build_dsv41_mm(layer.wo_a, out);
    cb(oa, "attn_wo_a", il);

    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_deepseek41::graph::build_candidate_mask(
        ggml_tensor * index_score,
        ggml_tensor * candidate_pin,
        int il) const {
    if (!candidate_pin) {
        // Every block fits in level one, so it cannot remove anything beyond causality.
        return nullptr;
    }

    const int64_t n_pos       = index_score->ne[0];
    const int64_t block_size  = hparams.dsv41_candidate_block_size;
    const int64_t n_blocks    = (n_pos + block_size - 1)/block_size;
    const int64_t n_pos_pad   = n_blocks*block_size;
    const int64_t n_pad       = n_pos_pad - n_pos;
    const int64_t top_blocks  = hparams.dsv41_candidate_top_k_blocks;

    GGML_ASSERT(block_size > 0 && n_blocks > top_blocks);
    GGML_ASSERT(candidate_pin->ne[0] == n_blocks);
    GGML_ASSERT(candidate_pin->ne[1] == index_score->ne[1]);
    GGML_ASSERT(candidate_pin->ne[2] == index_score->ne[2]);
    GGML_ASSERT(candidate_pin->ne[3] == index_score->ne[3]);

    // Pad the final partial block with -inf, exactly like the reference, before max pooling.
    ggml_tensor * padded = ggml_cont(ctx0, index_score);
    if (n_pad > 0) {
        ggml_tensor * pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                n_pad, index_score->ne[1], index_score->ne[2], index_score->ne[3]);
        pad = ggml_fill(ctx0, pad, -INFINITY);
        padded = ggml_concat(ctx0, padded, pad, 0);
    }

    ggml_tensor * block_score = ggml_pool_2d(ctx0, padded, GGML_OP_POOL_MAX,
            (int) block_size, 1, (int) block_size, 1, 0, 0);
    cb(block_score, "candidate_block_score", il);

    // The newest visible row may sit in a partial block; pin that block before top-k.
    block_score = ggml_add(ctx0, block_score, candidate_pin);
    ggml_tensor * top = ggml_cont(ctx0, ggml_top_k(ctx0, block_score, (int) top_blocks));

    // Scatter 0 into selected blocks and keep -inf everywhere else.
    ggml_tensor * keep = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
            1, n_blocks, block_score->ne[1], block_score->ne[3]);
    keep = ggml_fill(ctx0, keep, -INFINITY);

    ggml_tensor * top3 = ggml_view_4d(ctx0, top,
            top->ne[0], top->ne[1], top->ne[3], 1,
            top->nb[1], top->nb[2], top->ne[3]*top->nb[3], 0);
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
            1, top3->ne[0], top3->ne[1], top3->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    keep = ggml_set_rows(ctx0, keep, zeros, top3);
    keep = ggml_view_4d(ctx0, keep,
            keep->ne[1], keep->ne[2], 1, keep->ne[3],
            keep->nb[2], keep->nb[3], keep->nb[3], 0);
    cb(keep, "candidate_keep", il);

    // Expand each selected block back to rows, then trim only the padding added above.
    keep = ggml_reshape_4d(ctx0, keep, 1, n_blocks, keep->ne[1], keep->ne[3]);
    keep = ggml_repeat_4d(ctx0, keep, block_size, n_blocks, keep->ne[2], keep->ne[3]);
    keep = ggml_reshape_4d(ctx0, keep, n_pos_pad, keep->ne[2], 1, keep->ne[3]);
    if (n_pad > 0) {
        keep = ggml_view_4d(ctx0, keep, n_pos, keep->ne[1], keep->ne[2], keep->ne[3],
                keep->nb[1], keep->nb[2], keep->nb[3], 0);
    }
    cb(keep, "candidate_mask", il);

    return keep;
}

// Score this layer's queries against the shared index keys and keep the best compressed positions.
// The keys were published by an earlier layer, so this only builds the query side.
ggml_tensor * llama_model_deepseek41::graph::build_indexer_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * & candidate_mask_carry,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head      = hparams.indexer_n_head;
    const int64_t n_idx_head_dim  = hparams.indexer_head_size;
    const int64_t n_idx_head_rope = hparams.n_rot();
    const int64_t n_idx_head_nope = n_idx_head_dim - n_idx_head_rope;
    const int64_t nt              = cur->ne[1];

    GGML_ASSERT(inp_comp.kq_mask);
    GGML_ASSERT(n_idx_head_dim >= n_idx_head_rope);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_head_dim, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_idx_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    idx_q = ggml_rope_set_offset(idx_q, n_idx_head_nope);
    cb(idx_q, "idx_q", il);

    ggml_tensor * idx_k_rot = inp_dsv4->get_lid().k_rot;
    if (idx_k_rot) {
        idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);
        cb(idx_q, "idx_q_rot", il);
    }

    // one weight per head, scaled so the score matches the reference's
    // softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head)));
    cb(idx_w, "idx_weights", il);

    ggml_tensor * idx_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "idx_k", il);

    const int64_t n_stream = idx_k->ne[3];
    idx_q = ggml_view_4d(ctx0, idx_q,
            idx_q->ne[0], idx_q->ne[1], idx_q->ne[2]/n_stream, n_stream,
            idx_q->nb[1], idx_q->nb[2], idx_q->nb[3]/n_stream, 0);
    idx_w = ggml_view_4d(ctx0, idx_w,
            idx_w->ne[0], idx_w->ne[1]/n_stream, idx_w->ne[2], n_stream,
            idx_w->nb[1], idx_w->nb[2]/n_stream, idx_w->nb[3]/n_stream, 0);

    ggml_tensor * mask = inp_comp.kq_mask;

    const bool mask_candidates = nt == 1 && n_stream == 1 && cparams.fused_lid &&
        mask->type == GGML_TYPE_F16 && il > hparams.dsv41_candidate_source_layer &&
        candidate_mask_carry && dsv41_default_on("LLAMA_DSV41_INDEXER_MASK");
    if (mask_candidates) {
        // Candidate entries are exactly zero or -inf. Apply them before scoring.
        mask = ggml_add(ctx0, mask, candidate_mask_carry);
    }

    llm_sparse_selection_desc selection_desc;
    selection_desc.name               = "deepseek41_compressed";
    selection_desc.unit               = llm_sparse_selection_unit::compressed_entry;
    selection_desc.q_head_count       = (uint32_t) n_idx_head;
    selection_desc.k_head_count       = 1;
    selection_desc.head_dim           = (uint32_t) n_idx_head_dim;
    selection_desc.score_transform    = llm_sparse_score_transform::relu;
    selection_desc.score_reduction    = llm_sparse_score_reduction::weighted_head_sum;
    selection_desc.score_scale        = 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head));
    selection_desc.requested_top_k    = hparams.indexer_top_k;
    selection_desc.causal_policy      = llm_sparse_causal_policy::mask;
    selection_desc.tail_policy        = llm_sparse_tail_policy::local_window;
    selection_desc.expansion_policy   = llm_sparse_expansion_policy::mask;
    selection_desc.owner_layer        = il;
    selection_desc.source_layer       = hparams.dsv41_topk_source[il];
    selection_desc.reuse_allowed      = true;
    selection_desc.cache_location     = llm_sparse_cache_location::device;
    selection_desc.allowed_transports = LLM_SPARSE_TRANSPORT_SAME_DEVICE |
            LLM_SPARSE_TRANSPORT_P2P | LLM_SPARSE_TRANSPORT_HOST_STAGING;

    ggml_tensor * score = nullptr;
    if (cparams.fused_lid && mask->type == GGML_TYPE_F16) {
        score = ggml_lightning_indexer(ctx0, idx_q, idx_k, idx_w, mask);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
        idx_k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

        score = build_dsv41_mm(idx_k, idx_q);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));
        score = ggml_relu(ctx0, score);
        score = ggml_mul(ctx0, score, idx_w);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        if (mask->type != score->type) {
            mask = ggml_cast(ctx0, mask, score->type);
        }
        score = ggml_add(ctx0, score, mask);
    }
    cb(score, "idx_score", il);

    if (il == hparams.dsv41_candidate_source_layer) {
        candidate_mask_carry = build_candidate_mask(score, inp_comp.candidate_pin, il);
    } else if (il > hparams.dsv41_candidate_source_layer && candidate_mask_carry) {
        if (!mask_candidates) {
            score = ggml_add(ctx0, score, candidate_mask_carry);
        }
        cb(score, "idx_score_candidates", il);
    }

    llm_sparse_selection_request selection_request;
    selection_request.available_units = score->ne[0];
    const auto selection = llm_sparse_selection_dispatch(selection_desc, selection_request);
    llm_sparse_selection_profile(selection_desc, selection_request, selection);

    const uint32_t n_top_k = (uint32_t) selection.effective_top_k;

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
    cb(top_k, "idx_top_k", il);

    if (cparams.flash_attn && cparams.causal_attn && hparams.n_embd_head_k() == 512 && hparams.n_head(il) == 64 &&
            inp_dsv4->mctx->get_csa()->get_k(ctx0, il)->type == GGML_TYPE_F16 &&
            dsv41_compact_decode(nt, n_stream, n_comp, n_top_k, inp_dsv4->get_raw()->self_kv_order->ne[0], hparams.n_swa)) {
        // Keep the chronological order used by sparse attention mask compaction.
        top_k = dsv41_order_indices(ctx0, top_k);
        cb(top_k, "idx_top_k_ordered", il);
    }

    return top_k;
}

// DeepSeek-V4.1 attention: a sliding window of raw KV, plus, where the layer uses one, the
// compressed positions the indexer picked, concatenated into a single masked attention.
//
// Only a source layer compresses. The layers after it read the same rows, which the KV cache
// hands them through the reuse callback, so a reader builds no compressor at all.
ggml_tensor * llama_model_deepseek41::graph::build_attention_v41(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * & top_k_carry,
        ggml_tensor * & candidate_mask_carry,
        int il) {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const dsv41_rope_cfg rc = rope_cfg(il);

    // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    // the sliding window KV, which every layer keeps for itself
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = nullptr;

    if (ratio == 0) {
        // no compressed stream, so this layer sees only its own window
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);

        return build_attention_tail(model, out, inp_pos, nt, il);
    }

    // The plan slot follows the ratio, since a plan encodes how many tokens make a row. The rows
    // themselves always live in the CSA cache, and the index keys in the indexer cache, whichever
    // plan produced them.
    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    const auto & inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();

    const llama_dsv4_comp_state * comp_state = use_csa
        ? inp_dsv4->mctx->get_csa_state()
        : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(inp_comp.kq_mask && "a compressed layer needs a plan for its ratio");

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        const auto write_cache = [&](const auto * cache, ggml_tensor * value, uint32_t il_cache) {
            auto * write = cache->cpy_k(ctx0, value, inp_comp.state_write_idxs, il_cache);
            if (defer_cache_updates && il_cache > (uint32_t) il && model.dev_layer(il_cache) != model.dev_layer(il)) {
                cb(write, "deferred_cache_update", il_cache);
                cache_updates[il_cache].push_back(write);
            } else {
                ggml_build_forward_expand(gf, write);
            }
        };
        ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(state_kv, "comp_state_kv", il);

        // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below
        // then runs over a single element and returns 1.0 whatever the score holds, so the
        // values reach the cache unweighted, which is what a plain projection means.
        ggml_tensor * state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, cur)
            : state_kv;
        cb(state_score, "comp_state_score", il);

        const dsv4_state_tensors restored = dsv4_build_state_restore(ctx0, inp_comp, comp_state, il);

        ggml_tensor * base_kv = dsv4_view_2d(
                ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
        ggml_tensor * base_score = dsv4_view_2d(
                ctx0, restored.score, restored.score->ne[0], comp_state->get_n_rows(), 0);

        ggml_tensor * source_kv    = ggml_concat(ctx0, base_kv,    state_kv,    1);
        ggml_tensor * source_score = ggml_concat(ctx0, base_score, state_score, 1);

        // the indexer reads the latent before it is rotated, so ask for both forms at once
        ggml_tensor * latent_pre = nullptr;

        ggml_tensor * latent = build_hca_compressed_kv_from_state(
                source_kv,
                source_score,
                inp_comp.state_read_idxs,
                inp_comp.state_write_pos,
                layer.attn_comp_norm,
                ratio,
                n_embd_head,
                "comp_kv",
                il,
                &latent_pre);

        if (hparams.dsv41_owns_index_k(il)) {
            const int64_t n_idx_head_dim  = hparams.indexer_head_size;
            const int64_t n_idx_head_rope = hparams.n_rot();

            ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, latent_pre);
            idx_k = build_norm(idx_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
            idx_k = ggml_rope_ext(ctx0, idx_k, inp_comp.state_write_pos, nullptr, n_idx_head_rope,
                    rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                    dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
            idx_k = ggml_rope_set_offset(idx_k, n_idx_head_dim - n_idx_head_rope);
            cb(idx_k, "idx_k_new", il);

            if (inp_dsv4->get_lid().k_rot) {
                idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);
            }

            for (uint32_t il_cache : inp_dsv4->mctx->get_lid()->get_layer_ids()) {
                if (hparams.dsv41_index_key_source[il_cache] == (int32_t) il) {
                    write_cache(inp_dsv4->mctx->get_lid(), idx_k, il_cache);
                }
            }
        }

        if (inp_dsv4->get_csa().k_rot) {
            latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
            cb(latent, "comp_kv_rot", il);
        }

        // A V4.1 source layer can feed attention layers placed on several
        // devices.  When the CSA cache owns local reader replicas, copy only
        // the newly compressed rows to each replica here.  With shared source
        // storage the context exposes just the source layer, preserving the
        // original single-write graph and providing a runtime A/B fallback.
        for (uint32_t il_cache : inp_dsv4->mctx->get_csa()->get_layer_ids()) {
            if (hparams.dsv41_kv_source[il_cache] != (int32_t) il) {
                continue;
            }

            write_cache(inp_dsv4->mctx->get_csa(), latent, il_cache);
        }

        // carry whatever did not complete a row into the next ubatch
        ggml_tensor * snapshot_kv    = ggml_concat(ctx0, restored.kv,    state_kv,    1);
        ggml_tensor * snapshot_score = ggml_concat(ctx0, restored.score, state_score, 1);

        const dsv4_state_tensors snapshot = dsv4_build_state_snapshot(
                ctx0, inp_comp, comp_state, snapshot_kv, snapshot_score, il);
        if (snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, snapshot.kv);
        }
        if (snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, snapshot.score);
        }

        ggml_tensor * persist_kv = ggml_get_rows(ctx0, state_kv, inp_comp.state_persist_src_idxs);
        ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp_comp.state_persist_src_idxs);

        ggml_build_forward_expand(gf, comp_state->cpy_kv(
                    ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
        ggml_build_forward_expand(gf, comp_state->cpy_score(
                    ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
    }

    // an index source picks the positions; the layers in between reuse what it picked
    if (hparams.dsv41_is_index_source(il)) {
        selected_kv.clear();
        top_k_carry = build_indexer_top_k(
                model, inp_dsv4, inp_comp, qr, cur, inp_pos, candidate_mask_carry, il);
    }
    GGML_ASSERT(top_k_carry && "a compressed layer needs top-k from an index source");

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "comp_k", il);

    const bool compact = cparams.flash_attn && cparams.causal_attn && q->ne[0] == 512 && q->ne[1] == 64 &&
        comp_k->type == GGML_TYPE_F16 &&
        dsv41_compact_decode(nt, comp_k->ne[3], n_comp, top_k_carry->ne[0], inp_attn->self_kv_order->ne[0], hparams.n_swa);
    bool prefill = false;
    const char * prefill_reason = "disabled";
    if (dsv41_default_on("LLAMA_DSV41_PREFILL_COMPACT")) {
        const int64_t n_raw = inp_attn->self_kv_order ? inp_attn->self_kv_order->ne[0] : 0;
        if (nt == 1) {
            prefill_reason = "decode";
        } else if (is_dsv41_decode() || !inp_attn->queries.empty()) {
            prefill_reason = "verification";
        } else if (n_comp < dsv41_compact_min_compressed) {
            prefill_reason = "context";
        } else if (!cparams.flash_attn || !cparams.causal_attn || !cparams.offload_kqv || cparams.n_seq_max != 1 ||
                model.split_mode() != LLAMA_SPLIT_MODE_LAYER || comp_k->ne[3] != 1 || nt > 512 ||
                comp_k->type != GGML_TYPE_F16 || raw_k->type != GGML_TYPE_F16 ||
                n_raw < 1 || n_raw > 1024 || n_comp + n_raw >= (1LL << 24) || hparams.n_swa != 128 ||
                q->ne[0] != 512 || q->ne[1] != 64 || comp_k->ne[1] != 1 ||
                top_k_carry->ne[0] != 2048 || !ggml_is_matrix(top_k_carry) || !ggml_is_contiguous(top_k_carry) ||
                inp_comp.kq_mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(inp_comp.kq_mask) ||
                hparams.f_max_alibi_bias != 0.0f || hparams.attn_soft_cap) {
            prefill_reason = "mode-or-shape";
        } else {
            // Probe both operations before replacing the portable mask graph.
            auto * raw_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_raw, nt);
            auto * packed = ggml_dsv41_attn_indices(ctx0, raw_mask, inp_comp.kq_mask, top_k_carry, hparams.n_swa);
            const int64_t cap = top_k_carry->ne[0] + hparams.n_swa;
            auto * indices = ggml_view_2d(ctx0, packed, cap, nt, cap*sizeof(int32_t), 0);
            auto * counts = ggml_view_1d(ctx0, packed, nt, cap*nt*sizeof(int32_t));
            auto * kv_probe = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 512, n_raw + n_comp, 1);
            auto * fa_probe = ggml_flash_attn_ext(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3), kv_probe, kv_probe, nullptr, kq_scale, 0, 0);
            ggml_flash_attn_ext_set_indices(fa_probe, indices);
            ggml_flash_attn_ext_set_indices_counts(fa_probe, counts);
            const auto dev = model.dev_layer(il);
            prefill = ggml_backend_dev_supports_op(dev, packed) && ggml_backend_dev_supports_op(dev, fa_probe);
            prefill_reason = prefill ? "active" : "backend";
        }
    }
    if (dsv41_opt_in("LLAMA_DSV41_PREFILL_TRACE")) {
        LLAMA_LOG_INFO("dsv41 prefill compact: layer=%d queries=%" PRId64 " n_comp=%" PRId64 " %s\n", il, nt, n_comp, prefill_reason);
    }
    bool verify_compact = is_dsv41_verify_batch() && dsv41_default_on("LLAMA_DSV41_VERIFY_COMPACT") &&
        n_comp >= dsv41_compact_min_compressed && n_comp < (1LL << 24) - 1024 &&
        model.split_mode() == LLAMA_SPLIT_MODE_LAYER && raw_k->type == GGML_TYPE_F16 &&
        comp_k->type == GGML_TYPE_F16 && raw_k->ne[3] == 1 && comp_k->ne[3] == 1 &&
        inp_attn->queries.size() == (size_t) nt && inp_comp.query_n_kv.size() == (size_t) nt &&
        hparams.n_swa == 128 && q->ne[0] == 512 && q->ne[1] == 64 &&
        top_k_carry->ne[0] == 2048 && ggml_is_matrix(top_k_carry) && ggml_is_contiguous(top_k_carry) &&
        inp_comp.kq_mask->type == GGML_TYPE_F16 && ggml_is_contiguous(inp_comp.kq_mask) &&
        hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap;
    const int64_t verify_raw = inp_attn->self_kv_order ? inp_attn->self_kv_order->ne[0] : 0;
    for (size_t i = 0; verify_compact && i < inp_attn->queries.size(); ++i) {
        verify_compact = verify_raw > 0 && verify_raw <= 1024 &&
            inp_attn->queries[i]->self_kv_order->ne[0] == verify_raw && inp_comp.query_n_kv[i] == n_comp;
    }
    if (verify_compact) {
        auto * raw_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, verify_raw, nt);
        auto * packed = ggml_dsv41_attn_indices(ctx0, raw_mask, inp_comp.kq_mask, top_k_carry, hparams.n_swa);
        const int64_t cap = top_k_carry->ne[0] + hparams.n_swa;
        auto * indices = ggml_view_3d(ctx0, packed, cap, 1, nt, cap*sizeof(int32_t), cap*sizeof(int32_t), 0);
        auto * counts = ggml_view_2d(ctx0, packed, 1, nt, sizeof(int32_t), cap*nt*sizeof(int32_t));
        auto * kv_probe = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, 512, verify_raw + n_comp, 1, nt);
        auto * q_probe = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 512, 1, 64, nt);
        auto * fa_probe = ggml_flash_attn_ext(ctx0, q_probe, kv_probe, kv_probe, nullptr, kq_scale, 0, 0);
        ggml_flash_attn_ext_set_indices(fa_probe, indices);
        ggml_flash_attn_ext_set_indices_counts(fa_probe, counts);
        auto * mask_probe = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, verify_raw + n_comp, 1, 1, nt);
        auto * reference_probe = ggml_flash_attn_ext(ctx0, q_probe, kv_probe, kv_probe, mask_probe, kq_scale, 0, 0);
        ggml_flash_attn_ext_set_n_kv_max(reference_probe, cap);
        const auto dev = model.dev_layer(il);
        verify_compact = ggml_backend_dev_supports_op(dev, packed) && ggml_backend_dev_supports_op(dev, fa_probe) &&
            ggml_backend_dev_supports_op(dev, reference_probe);
    }
    ggml_tensor * comp_mask = compact || prefill || verify_compact ? inp_comp.kq_mask :
        build_top_k_mask(inp_comp.kq_mask, top_k_carry, "comp_top_k_mask", il);
    if (compact) {
        // Reuse selected rows while both the storage and Top-K source stay the same.
        const auto * storage = comp_k->view_src;
        GGML_ASSERT(storage);
        auto & selected = selected_kv[storage];
        if (!selected.first) {
            dsv41_select_kv(ctx0, comp_k, comp_mask, top_k_carry,
                    inp_attn->self_kv_order->ne[0], hparams.n_swa);
            cb(comp_k, "comp_k_selected", il);
            cb(comp_mask, "comp_mask_selected", il);
            selected = {comp_k, comp_mask};
        } else {
            comp_k = selected.first;
            comp_mask = selected.second;
        }
    }
    out = build_dsv41_attn(inp_attn, q, raw_k, comp_k, comp_mask, inp_comp.query_n_kv,
            layer.attn_sinks, top_k_carry->ne[0], kq_scale, il, prefill ? top_k_carry : nullptr,
            verify_compact ? top_k_carry : nullptr);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(model, out, inp_pos, nt, il);
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(pre_mix, "hc_pre_init", -1);

    ggml_tensor * top_k_carry = nullptr;
    ggml_tensor * candidate_mask_carry = nullptr;
    const bool is_media = ubatch.embd != nullptr || ubatch.embd_tensor != nullptr;
    defer_cache_updates = dsv41_default_on("LLAMA_DSV41_DEFER_CACHE_UPDATES") && is_dsv41_decode() &&
        cparams.n_seq_max == 1 && cparams.offload_kqv && model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
        model.n_gpu_layers() > hparams.n_layer_all;

    for (int il = 0; il < n_layer; ++il) {
        if (defer_cache_updates && il > 0 && model.dev_layer(il) != model.dev_layer(il - 1)) {
            ggml_build_forward_expand(gf, inpL);
            ggml_build_forward_expand(gf, pre_mix);
        }
        // Update each remote replica before its first local reader.
        auto updates = cache_updates.find(il);
        if (updates != cache_updates.end()) {
            for (auto * write : updates->second) {
                ggml_build_forward_expand(gf, write);
            }
            cache_updates.erase(updates);
        }
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // the engram sits before the block and writes straight into the stream
        if (!is_media && pmodel.engram_index(il) >= 0) {
            inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);

        cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(
                model, inp_dsv4, cur, inp_pos, top_k_carry, candidate_mask_carry, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl if the input is from mtmd
        if (is_media && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        const auto moe = [&](ggml_tensor * x) {
            return build_moe_ffn(x,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        };
        ggml_tensor * moe_out = nullptr;
        // Short verification can share MoE dispatch without changing attention.
        const char * batch_moe = std::getenv("LLAMA_DSV41_VERIFY_MOE_BATCH");
        const bool batch_verify_moe = is_dsv41_verify_batch() ||
                (cur->ne[1] <= 3 && batch_moe && std::strcmp(batch_moe, "1") == 0);
        if (is_dsv41_decode() && cur->ne[1] > 1 && !batch_verify_moe) {
            for (int64_t i = 0; i < cur->ne[1]; ++i) {
                ggml_tensor * x = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], i*cur->nb[1]);
                ggml_tensor * y = moe(x);
                moe_out = moe_out ? ggml_concat(ctx0, moe_out, y, 1) : y;
            }
        } else {
            moe_out = moe(cur);
        }
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    GGML_ASSERT(cache_updates.empty());
    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    // This is what a learned hyper-connection head does in V4, which is why this model has none.
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_dsv41_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
