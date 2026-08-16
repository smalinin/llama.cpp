#include "models.h"
#include "llama-kv-cache-msa.h"
#include <cmath>
#include <cstdint>

// MiniMax-M3: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling),
// swigluoai activation, and MiniMax Sparse Attention (MSA). MTP is not in released model weights.
// MSA blocks are defined over token positions. The graph translates between position space (block
// selection) and cell space (K/V/indexer storage) via per-ubatch pos<->cell maps populated from llama_kv_cells

void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks);
    msa_p = { (int) hparams.indexer_block_size, (int) hparams.indexer_top_k, (int) hparams.indexer_local_blocks };

    GGML_ASSERT(hparams.indexer_block_size > 0); // avoid div by zero

    switch (hparams.n_layer()) {
        case 60: type = LLM_TYPE_428B_A23B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_minimax_m3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_ff_exp        = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        // per-head QK-norm: a single head_dim vector applied to every head
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            // leading dense layers
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            // routed experts
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            // shared expert
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);

            // indexer
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, hparams.indexer_head_size}, 0);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {hparams.indexer_head_size}, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {hparams.indexer_head_size}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_m3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

class llm_graph_input_msa : public llm_graph_input_i {
public:
    llm_graph_input_msa(const llama_kv_cache_msa_context * mctx, int blk, bool unique_maps) :
        mctx(mctx), blk(blk), unique_maps(unique_maps) {}

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_pos_slot(pos_cell, query_map, ubatch, -1, unique_maps);
    }

    // valid as long as the position map still matches the new ubatch/cache window
    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_new = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

        this->mctx = mctx_new;

        const int64_t n_ps = GGML_PAD((int64_t) mctx_new->get_n_pos(params.ubatch), blk);
        const bool unique_maps_new = params.cparams.kv_unified && params.ubatch.n_seqs_unq > 1;
        const int64_t n_streams = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
        const int64_t n_maps = unique_maps_new ? params.ubatch.n_seqs_unq : n_streams;

        bool res = true;

        res &= pos_cell->ne[0] == n_ps;
        res &= pos_cell->ne[1] == n_maps;
        res &= query_map->ne[0] == (unique_maps_new ? params.ubatch.n_tokens : 1);
        res &= query_map->ne[1] == n_streams;

        return res;
    }

    ggml_tensor * pos_cell = nullptr; // I32 [n_ps, n_maps] pos -> cell, -1 for unmapped positions
    ggml_tensor * query_map = nullptr; // I32 [n_tps or 1, ns] query -> position-map index

    const llama_kv_cache_msa_context * mctx;
    int blk;
    bool unique_maps;
};

llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const auto & mm = static_cast<const llama_model_minimax_m3 &>(model);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // ==========================================
    // TODO: avoid such kind of complexity in the model graphs

    // MSA requires the non-transposed V layout provided by flash attention. With unified KV,
    // sparse ops use per-sequence position-to-cell maps plus a compact query-to-map index.
    const bool fa_on       = cparams.flash_attn;
    const bool msa_enabled = fa_on;

    auto * inp_attn = build_attn_inp_kv_msa(msa_enabled);

    static bool warned_no_fa = false;
    if (!fa_on && !warned_no_fa) {
        LLAMA_LOG_WARN("%s: flash attention disabled; MSA requires it -> running DENSE attention "
                       "(output may be degraded). Enable flash attention for MSA.\n", __func__);
        warned_no_fa = true;
    }
    // ==========================================

    // hoisted per-graph MSA state (shared by every sparse layer)
    llm_graph_input_msa * msa = nullptr;
    ggml_tensor * msa_kqm = nullptr;
    int64_t n_ps = 0, nblk = 0, ns = 1, n_tps = 0;
    bool msa_select = false;
    const int     blk = mm.msa_p.blk;
    const int64_t Hd  = hparams.indexer_n_head;   // one indexer head per GQA group

    if (msa_enabled) {
        const auto * mctx_msa = static_cast<const llama_kv_cache_msa_context *>(mctx);

        msa_kqm = inp_attn->get_kq_mask();
        n_tps = msa_kqm->ne[1];        // tokens per stream
        ns    = msa_kqm->ne[3];        // streams in this ubatch
        GGML_ASSERT(msa_kqm->type == GGML_TYPE_F16 && "MSA requires the FA (f16) mask");
        GGML_ASSERT(n_tps*ns == n_tokens);

        // the position axis covers every position currently in the cache and is padded to whole blocks
        n_ps = GGML_PAD((int64_t) mctx_msa->get_n_pos(ubatch), blk);
        nblk = n_ps / blk;
        msa_select = nblk > mm.msa_p.topk_blocks;

        if (msa_select) {
            const bool unique_maps = cparams.kv_unified && ubatch.n_seqs_unq > 1;
            auto inp = std::make_unique<llm_graph_input_msa>(mctx_msa, blk, unique_maps);

            const int64_t n_maps = unique_maps ? ubatch.n_seqs_unq : ns;
            inp->pos_cell = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_ps, n_maps);
            ggml_set_input(inp->pos_cell);
            inp->query_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, unique_maps ? n_tps : 1, ns);
            ggml_set_input(inp->query_map);

            msa = (llm_graph_input_msa *) res->add_input(std::move(inp));
        }
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // per-head QK RMSNorm (weights already include Gemma's +1)
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // partial rotary: only the first n_rot dims are rotated
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            const bool is_msa_layer = msa_enabled && il >= (int) hparams.n_layer_dense_lead;
            const bool is_sparse    = is_msa_layer && msa_select;

            ggml_tensor * iq = nullptr;
            ggml_tensor * ik_kv = nullptr;
            const llama_kv_cache_context * mctx_cur = nullptr;

            if (is_msa_layer) {
                const int64_t n_idx_dim = hparams.indexer_head_size;   // 128

                ggml_tensor * ik = build_lora_mm(model.layers[il].index_k_proj, cur);
                ik = ggml_reshape_3d(ctx0, ik, n_idx_dim, 1, n_tokens);
                ik = build_norm(ik, model.layers[il].index_k_norm, NULL, LLM_NORM_RMS, il);
                ik = ggml_rope_ext(ctx0, ik, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                if (inp_attn->self_k_rot_idx) {
                    ik = llama_mul_mat_hadamard(ctx0, ik, inp_attn->self_k_rot_idx);
                }

                const auto * mctx_msa_l = static_cast<const llama_kv_cache_msa_context *>(mctx);
                mctx_cur = mctx_msa_l->get_base();
                const auto * mctx_idx = mctx_msa_l->get_idx();
                ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, ik, inp_attn->get_k_idxs_idx(), il));

                if (is_sparse) {
                    iq = build_lora_mm(model.layers[il].index_q_proj, cur);
                    iq = ggml_reshape_3d(ctx0, iq, n_idx_dim, Hd, n_tokens);
                    iq = build_norm(iq, model.layers[il].index_q_norm, NULL, LLM_NORM_RMS, il);  // +1 baked
                    iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                       freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                    if (inp_attn->self_k_rot_idx) {
                        iq = llama_mul_mat_hadamard(ctx0, iq, inp_attn->self_k_rot_idx);
                    }
                    ik_kv = mctx_idx->get_k(ctx0, il);
                }
            }

            if (!is_sparse) {
                cur = build_attn(inp_attn, model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                const int64_t n_idx_dim = hparams.indexer_head_size;   // 128

                if (inp_attn->self_k_rot) {
                    Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp_attn->self_k_rot);
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
                }

                // Main branch: store K/V, take cache views
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
                ggml_tensor * k = mctx_cur->get_k(ctx0, il);
                ggml_tensor * v = mctx_cur->get_v(ctx0, il);
                GGML_ASSERT(!(v->nb[1] > v->nb[2]) && "MSA assumes v_trans=false (FA on)");

                const int64_t D   = k->ne[0];
                const int64_t HKV = k->ne[1];
                GGML_ASSERT(HKV == Hd && "MSA: one indexer head per GQA group");
                GGML_ASSERT(k->ne[3] == ns);
                const int K = mm.msa_p.topk_blocks < (int) nblk ? mm.msa_p.topk_blocks : (int) nblk;

                const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

                // Keep the score intermediate at block granularity.
                ggml_tensor * iq4 = ggml_reshape_4d(ctx0, iq, n_idx_dim, Hd, n_tps, ns);
                ggml_tensor * qpos = ggml_reshape_2d(ctx0, inp_pos, n_tps, ns);
                ggml_tensor * idx = ggml_msa_block_top_k(
                        ctx0, iq4, ik_kv, msa->pos_cell, msa->query_map, qpos, msa_kqm, K, blk, mm.msa_p.local);
                cb(idx, "msa_idx", il);

                // CUDA dispatches single-token decode to a dedicated kernel and prefill to
                // the KV-outer kernel. Both consume the cache in-place without gather tensors.
                ggml_tensor * q4 = ggml_reshape_4d(ctx0, Qcur, D, n_head, n_tps, ns);
                cur = ggml_msa_sparse_attn(
                        ctx0, q4, k, v, idx, msa->pos_cell, msa->query_map, qpos, msa_kqm, blk, kq_scale);
                cur = ggml_reshape_2d(ctx0, cur, D*n_head, n_tokens);
                if (inp_attn->self_v_rot) {
                    cur = llama_mul_mat_hadamard(ctx0, cur, inp_attn->self_v_rot);
                }
                cb(cur, "kqv_out", il);
                if (model.layers[il].wo) {
                    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
                }
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // leading dense FFN (swigluoai)
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // routed experts (swigluoai MoE)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (swigluoai)
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
