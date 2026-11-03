from pathlib import Path
R=Path('/home/sergei/Github/llama.cpp')
def edit(name,old,new):
 p=R/name;s=p.read_text();assert s.count(old)==1,(name,s.count(old),old[:90]);p.write_text(s.replace(old,new))
edit('src/llama-context.h','    bool pipeline_inputs_async = false;','    bool pipeline_inputs_async = false;\n    bool dsv41_candidate_block_local = false;')
edit('src/llama-context.cpp','    LLAMA_LOG_INFO("%s: constructing llama_context\\n", __func__);','''    LLAMA_LOG_INFO("%s: constructing llama_context\\n", __func__);
    const char * block_local = std::getenv("LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL");
    dsv41_candidate_block_local = block_local && strcmp(block_local, "1") == 0;''')
edit('src/llama-graph.h','    int32_t dsv41_prefill_kv_only_layer = -1;','    int32_t dsv41_prefill_kv_only_layer = -1;\n    bool dsv41_candidate_block_local = false;')
edit('src/llama-graph.h','        if (disable_moe_cache != other.disable_moe_cache) { return false; }','''        if (disable_moe_cache != other.disable_moe_cache) { return false; }
        if (dsv41_candidate_block_local != other.dsv41_candidate_block_local) { return false; }''')
edit('src/llama-context.cpp','        /*.res         =*/ res,','''        /*.res         =*/ res,
        /*.disable_moe_cache =*/ false,
        /*.dsv41_prefill_kv_only_layer =*/ -1,
        /*.dsv41_candidate_block_local =*/ dsv41_candidate_block_local && model.arch == LLM_ARCH_DEEPSEEK41 &&
            model.hparams.dsv41_candidate_source_layer == 20 && model.hparams.dsv41_candidate_block_size == 8 &&
            model.hparams.dsv41_candidate_top_k_blocks == 2048 && model.hparams.indexer_top_k == 2048 &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER && cparams.fused_lid && cparams.offload_kqv &&
            cparams.causal_attn && cparams.n_seq_max == 1 && !cparams.cb_eval && !abort_callback && loras->empty() &&
            ubatch.n_tokens > 6 && ubatch.n_tokens <= 512 && ubatch.n_seqs_unq == 1 &&
            ubatch.token && !ubatch.embd && !ubatch.embd_tensor,''')
edit('src/models/models.h','''        ggml_tensor * build_candidate_mask(
                ggml_tensor * index_score,
                ggml_tensor * candidate_pin,
                int il) const;''','''        bool candidate_block_local = false;
        std::map<int, std::pair<ggml_tensor *, ggml_backend_t>> candidate_local_masks;

        ggml_tensor * build_candidate_mask(
                const llama_model & model,
                ggml_tensor * index_score,
                ggml_tensor * candidate_pin,
                int il);''')
# The caller is const, so the graph-local construction state is mutable just like graph metadata.
edit('src/models/models.h','''        bool candidate_block_local = false;
        std::map<int, std::pair<ggml_tensor *, ggml_backend_t>> candidate_local_masks;''','''        bool candidate_block_local = false;
        mutable std::map<int, std::pair<ggml_tensor *, ggml_backend_t>> candidate_local_masks;''')
edit('src/models/models.h','''                ggml_tensor * candidate_pin,
                int il);''','''                ggml_tensor * candidate_pin,
                int il) const;''')
edit('src/models/deepseek41.cpp','''ggml_tensor * llama_model_deepseek41::graph::build_candidate_mask(
        ggml_tensor * index_score,''','''ggml_tensor * llama_model_deepseek41::graph::build_candidate_mask(
        const llama_model & model,
        ggml_tensor * index_score,''')
edit('src/models/deepseek41.cpp','''    keep = ggml_fill(ctx0, keep, -INFINITY);

    ggml_tensor * top3''','''    keep = ggml_fill(ctx0, keep, -INFINITY);
    ggml_tensor * keep_fill = keep;

    ggml_tensor * top3''')
edit('src/models/deepseek41.cpp','''    keep = ggml_set_rows(ctx0, keep, zeros, top3);
    keep = ggml_view_4d''','''    keep = ggml_set_rows(ctx0, keep, zeros, top3);
    ggml_tensor * keep_rows = keep;
    keep = ggml_view_4d''')
edit('src/models/deepseek41.cpp','''    // Expand each selected block back to rows, then trim only the padding added above.
    keep = ggml_reshape_4d(ctx0, keep, 1, n_blocks, keep->ne[1], keep->ne[3]);
    keep = ggml_repeat_4d(ctx0, keep, block_size, n_blocks, keep->ne[2], keep->ne[3]);
    keep = ggml_reshape_4d(ctx0, keep, n_pos_pad, keep->ne[2], 1, keep->ne[3]);
    if (n_pad > 0) {
        keep = ggml_view_4d(ctx0, keep, n_pos, keep->ne[1], keep->ne[2], keep->ne[3],
                keep->nb[1], keep->nb[2], keep->nb[3], 0);
    }
    cb(keep, "candidate_mask", il);

    return keep;''','''    auto expand = [&]() {
        auto * rows = ggml_reshape_4d(ctx0, keep, 1, n_blocks, keep->ne[1], keep->ne[3]);
        rows = ggml_repeat_4d(ctx0, rows, block_size, n_blocks, rows->ne[2], rows->ne[3]);
        auto * repeat = rows;
        rows = ggml_reshape_4d(ctx0, rows, n_pos_pad, rows->ne[2], 1, rows->ne[3]);
        if (n_pad > 0) {
            rows = ggml_view_4d(ctx0, rows, n_pos, rows->ne[1], rows->ne[2], rows->ne[3],
                    rows->nb[1], rows->nb[2], rows->nb[3], 0);
        }
        return std::make_pair(rows, repeat);
    };

    auto legacy = expand();
    cb(legacy.first, "candidate_mask", il);
    if (candidate_block_local && index_score->op == GGML_OP_LIGHTNING_INDEXER && index_score->ne[2] == 1 &&
            index_score->ne[3] == 1 && ggml_is_contiguous(keep) && hparams.dsv4_compress_ratios[il] == 1) {
        auto backend_for = [&](ggml_backend_dev_t dev) -> ggml_backend_t {
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                auto backend = ggml_backend_sched_get_backend(sched, i);
                if (ggml_backend_get_device(backend) == dev && strncmp(ggml_backend_name(backend), "CUDA", 4) == 0) {
                    return backend;
                }
            }
            return nullptr;
        };
        auto * keys = index_score->src[1];
        while (keys && keys->view_src) {
            keys = keys->view_src;
        }
        auto producer = backend_for(model.dev_layer(il));
        bool supported = producer && keys && keys->buffer && !ggml_backend_buffer_is_host(keys->buffer) &&
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(keys->buffer)) == model.dev_layer(il);
        std::map<int, ggml_backend_t> consumers;
        std::map<ggml_backend_t, std::pair<ggml_tensor *, ggml_tensor *>> expansions;
        for (int i = il + 1; supported && i < n_layer; ++i) {
            if (!hparams.dsv41_is_index_source(i)) {
                continue;
            }
            auto backend = backend_for(model.dev_layer(i));
            const auto * weights = model.layers[i].indexer_attn_q_b;
            supported = backend && weights && weights->buffer && !ggml_backend_buffer_is_host(weights->buffer) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(weights->buffer)) == model.dev_layer(i) &&
                hparams.dsv41_kv_source[i] == il && hparams.dsv41_index_key_source[i] == il &&
                hparams.dsv4_compress_ratios[i] == 1;
            if (!supported) {
                break;
            }
            auto & local = expansions[backend];
            if (!local.first) {
                local = expand();
            }
            auto * add = ggml_add(ctx0, index_score, local.first);
            supported = ggml_backend_supports_op(backend, local.second) && ggml_backend_supports_op(backend, add) &&
                ggml_backend_supports_op(backend, index_score);
            consumers[i] = backend;
        }
        if (supported && !consumers.empty()) {
            auto selector = consumers.begin()->second;
            const std::vector<ggml_tensor *> nodes = {block_score, top->src[0], top, keep_fill, zeros, keep_rows};
            supported = std::all_of(nodes.begin(), nodes.end(), [&](ggml_tensor * node) {
                return ggml_backend_supports_op(selector, node);
            });
            if (supported) {
                for (auto * node : nodes) {
                    ggml_backend_sched_set_tensor_backend(sched, node, selector);
                }
                for (const auto & item : consumers) {
                    auto local = expansions.at(item.second);
                    cb(local.second, "candidate_local_repeat", item.first);
                    cb(local.first, "candidate_local_mask", item.first);
                    ggml_backend_sched_set_tensor_backend(sched, local.second, item.second);
                    candidate_local_masks[item.first] = {local.first, item.second};
                }
            }
        }
    }
    return legacy.first;''')
edit('src/models/deepseek41.cpp','''        candidate_mask_carry = build_candidate_mask(score, inp_comp.candidate_pin, il);''','''        candidate_mask_carry = build_candidate_mask(model, score, inp_comp.candidate_pin, il);''')
edit('src/models/deepseek41.cpp','''        if (!mask_candidates) {
            score = ggml_add(ctx0, score, candidate_mask_carry);
        }
        cb(score, "idx_score_candidates", il);''','''        if (!mask_candidates) {
            auto local = candidate_local_masks.find(il);
            if (local != candidate_local_masks.end()) {
                GGML_ASSERT(ggml_are_same_shape(score, local->second.first));
                ggml_backend_sched_set_tensor_backend(sched, score, local->second.second);
                score = ggml_add(ctx0, score, local->second.first);
                ggml_backend_sched_set_tensor_backend(sched, score, local->second.second);
            } else {
                score = ggml_add(ctx0, score, candidate_mask_carry);
            }
        }
        cb(score, "idx_score_candidates", il);''')
edit('src/models/deepseek41.cpp','''    prefill_kv_only_layer = params.dsv41_prefill_kv_only_layer;''','''    prefill_kv_only_layer = params.dsv41_prefill_kv_only_layer;
    candidate_block_local = params.dsv41_candidate_block_local;''')
