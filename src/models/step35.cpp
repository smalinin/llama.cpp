#include "models.h"

#include "llama-kv-cache-dsa-iswa.h"
#include "llama-sparse-selection.h"

#include <cstdlib>
#include <cstring>

void llama_model_step35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // Legacy Step35 uses half of head_dim; explicit GGUF dimensions take precedence.
    if (!ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false)) {
        hparams.n_rot_full = hparams.n_rot_full / 2;
    }

    has_sparse_indexer = ml.get_weight("blk.3.attn_ssmax_s") != nullptr;
    if (has_sparse_indexer) {
        const char * sparse_env = std::getenv("LLAMA_STEP5_EXPERIMENTAL_SPARSE");
        if (sparse_env && std::strcmp(sparse_env, "1") == 0) {
            hparams.indexer_n_head = 16;
            hparams.indexer_head_size = 256;
            // Use a larger token budget until Step-5 block selection is available.
            hparams.indexer_top_k = 4096;
            LLAMA_LOG_WARN("Step-5: experimental token indexer enabled, top-k=%u tokens; DSA scoring, no CSA compression or provider groups, long-context quality is unverified\n", hparams.indexer_top_k);
        }
        const char * env = std::getenv("LLAMA_STEP5_EXPERIMENTAL_DENSE");
        experimental_dense_attention = env != nullptr && std::strcmp(env, "1") == 0;
        if (experimental_dense_attention && hparams.indexer_top_k == 0) {
            LLAMA_LOG_WARN("Step-5: experimental dense long-context attention enabled; CSA selection is not implemented and long-context quality is unverified\n");
        } else if (hparams.indexer_top_k == 0) {
            LLAMA_LOG_WARN("Step-5: dense attention with context <= 512; set LLAMA_STEP5_EXPERIMENTAL_DENSE=1 to allow experimental long context\n");
        }
    }
    if (hparams.n_layer() == 92 && hparams.n_expert == 352 && !has_sparse_indexer) {
        throw std::runtime_error("Step-5 GGUF is missing sparse tensors; attach the sparse GGUF split");
    }

    // MoE + SWA parameters
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);

    // Step35 uses sigmoid gating by default (if not set in GGUF)
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,  hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,        hparams.rope_freq_base_train_swa, false);

    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer_all);

    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    switch (hparams.n_layer()) {
        case 45: type = LLM_TYPE_196B_A11B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_step35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    // Trunk-only: the GGUF declares MTP layers in metadata but the actual MTP
    // tensors live in a separate file (e.g. user split target/draft). Mark
    // MTP tensors NOT_REQUIRED so the trunk loads cleanly.
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only  ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags         = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, trunk_flags);

    // STEP35 supports per-layer partial RoPE dims; rope factors are stored as a single shared tensor
    // ("rope_freqs.weight") and ggml uses only the first (n_rot_l/2) entries per layer.
    uint32_t n_rot_max = 0;
    for (int i = 0; i < n_layer; ++i) {
        n_rot_max = std::max(n_rot_max, hparams.n_rot(i));
    }
    if (n_rot_max == 0) {
        n_rot_max = n_rot;
    }

    auto load_block_trunk = [&](int i, int flags) {
        auto & layer = layers[i];

        const uint32_t n_head_l      = hparams.n_head(i);
        const uint32_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
        const uint32_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        // optional rope factors (llama3) / longrope tensors
        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        } else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_l, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v * n_head_l, n_embd}, flags);

        // head-wise attention gate (Step35 self_attn.g_proj)
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_l}, TENSOR_NOT_REQUIRED);

        if (has_sparse_indexer && !hparams.is_swa(i)) {
            const int64_t proxy_dim = 256;
            const int64_t index_heads = 16;
            layer.index_q_proj   = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, proxy_dim * index_heads}, flags);
            layer.index_k_proj   = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, proxy_dim}, flags);
            layer.index_z_proj   = create_tensor(tn(LLM_TENSOR_INDEXER_Z_PROJ, "weight", i), {n_embd, proxy_dim}, flags);
            layer.indexer_proj   = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ, "weight", i), {n_embd, index_heads}, flags);
            layer.index_q_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {proxy_dim}, flags);
            layer.index_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {proxy_dim}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "bias", i), {proxy_dim}, flags);
            layer.attn_ssmax_s   = create_tensor(tn(LLM_TENSOR_ATTN_SSMAX_S, i), {n_head_l}, flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // dense MLP (leading dense blocks)
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);

        // MoE routed experts + selection bias (router_bias)
        const int64_t n_ff_exp = hparams.n_ff_exp();
        layer.ffn_gate_inp      = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps     = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps       = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_exp_probs_b   = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

        // shared expert MLP
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);
    };

    auto load_block_mtp = [&](int i) {
        auto & layer = layers[i];

        const uint32_t n_head_l      = hparams.n_head(i);
        const uint32_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
        const uint32_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);

        // The MTP block is a full Step3p5 decoder layer (mtp_block) plus the
        // NextN-specific wiring (enorm/hnorm/eh_proj + optional shared head).
        // Multi-block MTP: every declared MTP block is required (the draft chain
        // runs all n_layer_nextn heads), so each block uses the captured
        // `mtp_flags` directly — already NOT_REQUIRED for a trunk-only GGUF,
        // which keeps that path correct.

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | TENSOR_DUPLICATED);
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | TENSOR_DUPLICATED);
        } else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot_max/2}, TENSOR_NOT_REQUIRED | TENSOR_DUPLICATED);
        }

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_l, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v * n_head_l, n_embd}, mtp_flags);

        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_l}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, mtp_flags);

        // dense MLP (leading dense blocks) — present if the MTP block isn't MoE
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);

        // MoE routed experts + selection bias (router_bias)
        const int64_t n_ff_exp = hparams.n_ff_exp();
        layer.ffn_gate_inp      = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps     = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps       = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp,   n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_exp_probs_b   = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd },              TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    // All n_layer_nextn MTP blocks are required — the multi-block draft chain
    // runs every head (head k at offset k). The GGUF declares the count via
    // step35.nextn_predict_layers.
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_step35::build_arch_graph(const llm_graph_params & params) const {
    if (has_sparse_indexer && params.cparams.n_ctx > 512 && !experimental_dense_attention && hparams.indexer_top_k == 0) {
        throw std::runtime_error("Step-5 CSA selection is not implemented; set LLAMA_STEP5_EXPERIMENTAL_DENSE=1 to allow experimental dense attention beyond 512 tokens");
    }
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_step35::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos     = build_inp_pos();
    auto * inp_attn = hparams.indexer_top_k == 0 ? build_attn_inp_kv_iswa() : nullptr;
    auto * inp_sparse = hparams.indexer_top_k > 0 ? build_attn_inp_k_dsa_iswa() : nullptr;
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const uint32_t n_head_l    = hparams.n_head(il);
        const uint32_t n_head_kv_l = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        // dump pre-attn RMSNorm input to pinpoint layer boundary issues
        cb(cur, "attn_norm_in", il);

        // self-attention
        {
            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head_k, n_head_l,
                    n_embd_head_k, n_head_kv_l,
                    n_embd_head_v, n_head_kv_l,
                    il, false);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);

            // Q/K per-head RMSNorm (Step35 q_norm / k_norm)
            if (model.layers[il].attn_q_norm) {
                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);
            }

            // RoPE (partial rotary factors per layer)
            const bool is_swa = hparams.is_swa(il);
            ggml_tensor * rope_factors = is_swa ? nullptr : model.get_rope_factors(cparams, il);
            const int64_t n_rot_l = hparams.n_rot(il);
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
            );
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
            );
            cb(Qcur, "Qcur_pos", il);
            cb(Kcur, "Kcur_pos", il);

            float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
            if (model.layers[il].attn_ssmax_s) {
                // Experimental short-context scale reported by the checkpoint author.
                auto * scale = ggml_reshape_3d(ctx0, model.layers[il].attn_ssmax_s, 1, n_head_l, 1);
                Qcur = ggml_mul(ctx0, Qcur, scale);
                cb(Qcur, "Qcur_ssmax", il);
                kq_scale = 1.0f;
            }
            ggml_tensor * attn_out = inp_sparse
                ? build_token_indexer_attn(inp_sparse, cur, inp_pos, Qcur, Kcur, Vcur, kq_scale, il)
                : build_attn(inp_attn, nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(attn_out, "attn_out", il);
            // head-wise attention gate: sigmoid(g_proj(x)) in torch
            if (model.layers[il].wqkv_gate) {
                ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur); // [n_head_l, n_tokens]
                cb(gate, "attn_gate", il);

                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate_sigmoid", il);

                // reshape + broadcast to [n_embd_head_v, n_head_l, n_tokens]
                ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head_l, n_tokens);
                ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,       1,          n_head_l, n_tokens);
                cb(gate_3d, "attn_gate_3d", il);

                attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
                cb(attn_3d, "attn_gated_3d", il);

                attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v * n_head_l, n_tokens);
                cb(attn_out, "attn_gated", il);
            }

            // output projection
            cur = build_lora_mm(model.layers[il].wo, attn_out, model.layers[il].wo_s);
            cb(cur, "attn_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense MLP
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, nullptr,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE routed experts
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert MLP (always added on MoE layers in Step35)
            ggml_tensor * sh_out = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(sh_out, "ffn_shared_out", il);

            cur = ggml_add(ctx0, moe_out, sh_out);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_step35::graph::build_token_indexer_attn(
        llm_graph_input_attn_k_dsa_iswa * inp,
        ggml_tensor * hidden,
        ggml_tensor * pos,
        ggml_tensor * q,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        float scale,
        int il) {
    const bool is_swa = hparams.is_swa(il);
    const auto * cache = is_swa ? inp->mctx->get_swa() : inp->mctx->get_dsa()->get_mla();
    auto * idxs = is_swa ? inp->get_swa()->get_k_idxs() : inp->get_dsa()->get_k_idxs_mla();
    auto * mask = is_swa ? inp->get_swa()->get_kq_mask() : inp->get_dsa()->get_kq_mask_mla();

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, cache->cpy_k(ctx0, k_cur, idxs, il));
    ggml_build_forward_expand(gf, cache->cpy_v(ctx0, v_cur, idxs, il));

    auto * k = cache->get_k(ctx0, il);
    auto * v = cache->get_v(ctx0, il);
    if (is_swa) {
        return build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, 0, scale, il);
    }

    const auto & layer = model.layers[il];
    const int64_t dim = hparams.indexer_head_size;
    const int64_t heads = hparams.indexer_n_head;
    const auto * index_cache = inp->mctx->get_dsa()->get_lid();
    auto * index_k = build_lora_mm(layer.index_k_proj, hidden);
    index_k = build_norm(index_k, layer.index_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    index_k = ggml_reshape_3d(ctx0, index_k, dim, 1, n_tokens);
    index_k = ggml_rope_ext(ctx0, index_k, pos, nullptr, 32, rope_type, n_ctx_orig,
            model.get_rope_freq_base(cparams, il), model.get_rope_freq_scale(cparams, il),
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(index_k, "step5_index_k", il);
    ggml_build_forward_expand(gf, index_cache->cpy_k(ctx0, index_k, inp->get_dsa()->get_k_idxs_lid(), il));

    if (k->ne[2] <= hparams.indexer_top_k) {
        return build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, 0, scale, il);
    }

    auto * index_q = build_lora_mm(layer.index_q_proj, hidden);
    index_q = ggml_reshape_3d(ctx0, index_q, dim, heads, n_tokens);
    index_q = build_norm(index_q, layer.index_q_norm, nullptr, LLM_NORM_RMS, il);
    index_q = ggml_rope_ext(ctx0, index_q, pos, nullptr, 32, rope_type, n_ctx_orig,
            model.get_rope_freq_base(cparams, il), model.get_rope_freq_scale(cparams, il),
            ext_factor, attn_factor, beta_fast, beta_slow);
    auto * weights = build_lora_mm(layer.indexer_proj, hidden);
    weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(dim * heads)));

    index_k = index_cache->get_k(ctx0, il);
    const int64_t streams = index_k->ne[3];
    index_q = ggml_view_4d(ctx0, index_q, dim, heads, n_tokens / streams, streams,
            index_q->nb[1], index_q->nb[2], index_q->nb[3] / streams, 0);
    weights = ggml_view_4d(ctx0, weights, heads, n_tokens / streams, 1, streams,
            weights->nb[1], weights->nb[2], weights->nb[2] / streams, 0);
    index_q = ggml_permute(ctx0, index_q, 0, 2, 1, 3);
    index_k = ggml_permute(ctx0, index_k, 0, 2, 1, 3);
    auto * score = ggml_mul_mat(ctx0, index_k, index_q);
    ggml_mul_mat_set_prec(score, GGML_PREC_F32);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));
    score = ggml_mul(ctx0, ggml_relu(ctx0, score), weights);
    score = ggml_sum_rows(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));
    auto * index_mask = inp->get_dsa()->get_kq_mask_lid();
    score = ggml_add(ctx0, score, ggml_cast(ctx0, index_mask, GGML_TYPE_F32));
    cb(score, "step5_index_score", il);

    llm_sparse_selection_desc desc;
    desc.name = "step5_experimental_token";
    desc.q_head_count = heads;
    desc.k_head_count = 1;
    desc.head_dim = dim;
    desc.score_transform = llm_sparse_score_transform::relu;
    desc.score_reduction = llm_sparse_score_reduction::weighted_head_sum;
    desc.score_scale = 1.0f / sqrtf(float(dim * heads));
    desc.requested_top_k = hparams.indexer_top_k;
    desc.causal_policy = llm_sparse_causal_policy::mask;
    desc.expansion_policy = n_tokens == streams ? llm_sparse_expansion_policy::gather : llm_sparse_expansion_policy::mask;
    desc.owner_layer = il;
    desc.source_layer = il;
    desc.cache_location = llm_sparse_cache_location::device;
    desc.allowed_transports = LLM_SPARSE_TRANSPORT_SAME_DEVICE | LLM_SPARSE_TRANSPORT_P2P | LLM_SPARSE_TRANSPORT_HOST_STAGING;
    llm_sparse_selection_request request;
    request.available_units = k->ne[2];
    request.prefer_gather = n_tokens == streams;
    const auto selection = llm_sparse_selection_dispatch(desc, request);
    llm_sparse_selection_profile(desc, request, selection);
    auto * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, selection.effective_top_k));
    cb(top_k, "step5_top_k", il);

    if (n_tokens == streams) {
        // Decode gathers separate K and V rows for all four GQA key heads.
        auto * indices = ggml_reshape_3d(ctx0, top_k, top_k->ne[0], 1, streams);
        auto gather = [&](ggml_tensor * values) {
            auto * packed = ggml_view_4d(ctx0, values, values->ne[0] * values->ne[1], values->ne[2], 1, streams,
                    values->nb[2], values->nb[3], values->nb[3], 0);
            auto * rows = ggml_get_rows(ctx0, packed, indices);
            rows = ggml_reshape_4d(ctx0, rows, values->ne[0], values->ne[1], top_k->ne[0], streams);
            return ggml_cast(ctx0, rows, GGML_TYPE_F16);
        };
        k = gather(k);
        v = gather(v);
        auto * mask_rows = ggml_reshape_4d(ctx0, mask, 1, mask->ne[0], 1, streams);
        mask = ggml_get_rows(ctx0, mask_rows, indices);
        mask = ggml_reshape_4d(ctx0, mask, top_k->ne[0], 1, 1, streams);
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        cb(k, "step5_selected_k", il);
        cb(v, "step5_selected_v", il);
    } else {
        mask = build_attn_kq_mask_top_k(inp->get_dsa(), top_k, il);
    }
    const int64_t n_kv_max = n_tokens == streams ? 0 : top_k->ne[0];
    return build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, n_kv_max, scale, il);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Step3p5 (MoE)
llama_model_step35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "STEP35 MTP requires n_layer_nextn > 0");

    // Multi-block MTP: the DECODER_MTP graph runs the MTP head selected by
    // cparams.nextn_layer_offset (0 = first trained head). The speculative driver
    // bumps the offset per draft step to chain heads 45->46->47. offset 0 keeps
    // single-block behavior identical to before.
    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    const uint32_t n_head_l    = hparams.n_head(il);
    const uint32_t n_head_kv_l = hparams.n_head_kv(il);

    const float freq_base_l  = model.get_rope_freq_base(cparams, il);
    const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_input = inp->build_h(ctx0, ubatch);
    ggml_set_name(h_input, "mtp_h_input");

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos  = build_inp_pos();
    auto        * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    // mtp_block: full Step3p5 decoder layer (attention with optional head-wise gate, then MoE/dense FFN)
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head_k, n_head_l,
            n_embd_head_k, n_head_kv_l,
            n_embd_head_v, n_head_kv_l,
            il, false);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l,    n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv_l, n_tokens);

    if (layer.attn_q_norm) {
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);
    }
    if (layer.attn_k_norm) {
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "mtp_Kcur_normed", il);
    }

    const bool    is_swa       = hparams.is_swa(il);
    ggml_tensor * rope_factors = is_swa ? nullptr : model.get_rope_factors(cparams, il);
    const int64_t n_rot_l      = hparams.n_rot(il);

    Qcur = ggml_rope_ext(
        ctx0, Qcur, inp_pos, rope_factors,
        n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
        ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(
        ctx0, Kcur, inp_pos, rope_factors,
        n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
        ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur_pos", il);
    cb(Kcur, "mtp_Kcur_pos", il);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
    ggml_tensor * attn_out = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(attn_out, "mtp_attn_out", il);

    // head-wise attention gate: sigmoid(g_proj(x))
    if (layer.wqkv_gate) {
        ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur); // [n_head_l, n_tokens]
        cb(gate, "mtp_attn_gate", il);

        gate = ggml_sigmoid(ctx0, gate);
        cb(gate, "mtp_attn_gate_sigmoid", il);

        ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head_l, n_tokens);
        ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,       1,           n_head_l, n_tokens);
        cb(gate_3d, "mtp_attn_gate_3d", il);

        attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
        cb(attn_3d, "mtp_attn_gated_3d", il);

        attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head_v * n_head_l, n_tokens);
        cb(attn_out, "mtp_attn_gated", il);
    }

    cur = build_lora_mm(layer.wo, attn_out, layer.wo_s);
    cb(cur, "mtp_attn_proj", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_inp = cur;
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    // FFN: dense MLP or MoE (mirrors trunk path)
    if (layer.ffn_gate_inp == nullptr) {
        cur = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   nullptr,
                layer.ffn_gate, layer.ffn_gate_b, nullptr,
                layer.ffn_down, layer.ffn_down_b, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "mtp_ffn_out", il);
    } else {
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "mtp_ffn_moe_out", il);

        ggml_tensor * sh_out = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(sh_out, "mtp_ffn_shared_out", il);

        cur = ggml_add(ctx0, moe_out, sh_out);
        cb(cur, "mtp_ffn_out", il);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    // Pre-norm hidden state: used by the AR draft loop to seed the next MTP step.
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "STEP35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "STEP35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
