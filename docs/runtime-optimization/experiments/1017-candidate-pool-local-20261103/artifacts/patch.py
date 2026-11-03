from pathlib import Path
R=Path('/home/sergei/Github/llama.cpp')
p=R/'src/models/deepseek41.cpp';s=p.read_text();s=s.replace('ggml_tensor * padded = ggml_cont(ctx0, index_score);\n    if (n_pad > 0)', 'ggml_tensor * padded = ggml_cont(ctx0, index_score);\n    cb(padded, "candidate_score_cont", il);\n    if (n_pad > 0)');s=s.replace('pad = ggml_fill(ctx0, pad, -INFINITY);\n        padded = ggml_concat(ctx0, padded, pad, 0);', 'pad = ggml_fill(ctx0, pad, -INFINITY);\n        cb(pad, "candidate_score_pad", il);\n        padded = ggml_concat(ctx0, padded, pad, 0);\n        cb(padded, "candidate_score_padded", il);');s=s.replace('block_score = ggml_add(ctx0, block_score, candidate_pin);\n    ggml_tensor * top', 'block_score = ggml_add(ctx0, block_score, candidate_pin);\n    cb(block_score, "candidate_pinned_block_score", il);\n    ggml_tensor * top');p.write_text(s)
p=R/'src/llama-context.cpp';s=p.read_text();needle='''        if (sparse_profile) {
            const char * category = sparse_profile_category(name);''';assert s.count(needle)==1
block='''        if (model.arch == LLM_ARCH_DEEPSEEK41 && strcmp(name, "candidate_pinned_block_score") == 0) {
            const char * local = std::getenv("LLAMA_DSV41_CANDIDATE_POOL_LOCAL");
            if (local && strcmp(local, "1") == 0 && il == 20 &&
                    model.hparams.dsv41_candidate_source_layer == 20 &&
                    model.hparams.dsv41_candidate_block_size == 8 && model.hparams.dsv41_candidate_top_k_blocks == 2048 &&
                    model.hparams.indexer_top_k == 2048 && model.hparams.dsv4_compress_ratios[il] == 1 &&
                    model.split_mode() == LLAMA_SPLIT_MODE_LAYER && cparams.fused_lid && cparams.offload_kqv &&
                    cparams.causal_attn && cparams.n_seq_max == 1 && !cparams.cb_eval && !abort_callback && loras->empty() &&
                    ubatch.n_tokens > 6 && ubatch.n_tokens <= 512 && ubatch.n_seqs_unq == 1 &&
                    ubatch.token && !ubatch.embd && !ubatch.embd_tensor && cur->op == GGML_OP_ADD && cur->type == GGML_TYPE_F32) {
                auto * pool = cur->src[0];
                auto * input = pool->src[0];
                std::vector<ggml_tensor *> nodes = {pool};
                if (input->op == GGML_OP_CONCAT) {
                    nodes.push_back(input);
                    nodes.push_back(input->src[1]);
                    input = input->src[0];
                }
                nodes.push_back(input);
                auto * score = input->src[0];
                auto * keys = score ? score->src[1] : nullptr;
                while (keys && keys->view_src) {
                    keys = keys->view_src;
                }
                const auto owner = model.dev_layer(il);
                ggml_backend_t producer = nullptr;
                ggml_backend_t selector = nullptr;
                int next_source = il + 1;
                while (next_source < (int) model.hparams.n_layer() && !model.hparams.dsv41_is_index_source(next_source)) {
                    ++next_source;
                }
                if (next_source == 24 && pool->op == GGML_OP_POOL_2D && input->op == GGML_OP_CONT &&
                        score && score->op == GGML_OP_LIGHTNING_INDEXER && score->type == GGML_TYPE_F32 &&
                        score->ne[1] == ubatch.n_tokens && score->ne[2] == 1 && score->ne[3] == 1 && ggml_is_contiguous(score) &&
                        keys && keys->buffer && !ggml_backend_buffer_is_host(keys->buffer) &&
                        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(keys->buffer)) == owner) {
                    nodes.push_back(score);
                    for (const auto & backend : backends) {
                        const char * backend_name = ggml_backend_name(backend.get());
                        if (strncmp(backend_name, "CUDA", 4) != 0) {
                            continue;
                        }
                        if (ggml_backend_get_device(backend.get()) == owner &&
                                std::all_of(nodes.begin(), nodes.end(), [&](ggml_tensor * node) {
                                    return node && ggml_backend_supports_op(backend.get(), node);
                                })) {
                            producer = backend.get();
                        }
                        if (ggml_backend_get_device(backend.get()) == model.dev_layer(next_source) &&
                                ggml_backend_supports_op(backend.get(), cur)) {
                            selector = backend.get();
                        }
                    }
                }
                if (producer && selector) {
                    for (auto * node : nodes) {
                        ggml_backend_sched_set_tensor_backend(sched.get(), node, producer);
                    }
                    ggml_backend_sched_set_tensor_backend(sched.get(), cur, selector);
                }
            }
        }

'''
s=s.replace(needle,block+needle);p.write_text(s)
