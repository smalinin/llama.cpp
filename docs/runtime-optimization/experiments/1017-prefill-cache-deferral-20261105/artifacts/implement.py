from pathlib import Path
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105';P=E/'artifacts';P.mkdir(parents=True)
f=R/'src/llama-kv-cache-dsv4.h';s=f.read_text();needle='    std::vector<uint32_t> get_layer_ids() const;\n\n    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;';assert s.count(needle)==1;s=s.replace(needle,'    std::vector<uint32_t> get_layer_ids() const;\n    ggml_type type_k() const;\n\n    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;');f.write_text(s)
f=R/'src/llama-kv-cache-dsv4.cpp';s=f.read_text();needle='ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k(ggml_context * ctx, int32_t il) const {';assert s.count(needle)==1;s=s.replace(needle,'ggml_type llama_kv_cache_dsv4_comp_context::type_k() const {\n    return kv->type_k();\n}\n\n'+needle);f.write_text(s)
f=R/'src/models/deepseek41.cpp';s=f.read_text();old='''            if (defer_cache_updates && il_cache > (uint32_t) il && model.dev_layer(il_cache) != model.dev_layer(il)) {
                cb(write, "deferred_cache_update", il_cache);''';new='''            const auto * storage = write->view_src;
            const bool known_owner = is_dsv41_decode() || (storage && storage->buffer &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(storage->buffer)) == model.dev_layer(il_cache));
            if (defer_cache_updates && known_owner && il_cache > (uint32_t) il && model.dev_layer(il_cache) != model.dev_layer(il)) {
                cb(write, "deferred_cache_update", il_cache);''';assert s.count(old)==1;s=s.replace(old,new)
old='''    defer_cache_updates = dsv41_default_on("LLAMA_DSV41_DEFER_CACHE_UPDATES") && is_dsv41_decode() &&
        cparams.n_seq_max == 1 && cparams.offload_kqv && model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
        model.n_gpu_layers() > hparams.n_layer_all;

    for (int il = 0; il < n_layer; ++il) {
        if (defer_cache_updates && il > 0 && model.dev_layer(il) != model.dev_layer(il - 1)) {'''
new='''    const char * prefill_defer = std::getenv("LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES");
    bool prefill_cache_updates = prefill_defer && std::strcmp(prefill_defer, "1") == 0 &&
        !is_dsv41_decode() && n_tokens > 1 && n_tokens <= 512 && ubatch.token && !is_media &&
        ubatch.n_seqs_unq == 1 && cparams.n_seq_max == 1 && !cparams.cb_eval &&
        cparams.causal_attn && cparams.flash_attn && cparams.offload_kqv && cparams.pipeline_parallel &&
        model.split_mode() == LLAMA_SPLIT_MODE_LAYER && model.n_gpu_layers() > hparams.n_layer_all &&
        inp_dsv4->mctx->get_csa()->type_k() == GGML_TYPE_Q8_0 &&
        inp_dsv4->mctx->get_lid()->type_k() == GGML_TYPE_Q8_0;
    if (prefill_cache_updates) {
        const auto on_cuda = [](const ggml_tensor * tensor) {
            if (!tensor) {
                return true;
            }
            if (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer)) {
                return false;
            }
            const auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            return dev && std::strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0;
        };
        for (int il = 0; il < n_layer && prefill_cache_updates; ++il) {
            const auto & layer = model.layers[il];
            for (const auto * tensor : {layer.wq_a, layer.wq_b, layer.wkv, layer.wo_a, layer.wo_b,
                    layer.attn_norm, layer.attn_q_a_norm, layer.attn_kv_norm, layer.attn_comp_wkv,
                    layer.attn_comp_wgate, layer.attn_comp_norm, layer.indexer_attn_k, layer.indexer_k_norm,
                    layer.ffn_gate_inp, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_gate_up_exps,
                    layer.ffn_down_exps, layer.ffn_gate_shexp, layer.ffn_up_shexp, layer.ffn_down_shexp}) {
                if (!on_cuda(tensor)) {
                    prefill_cache_updates = false;
                    break;
                }
            }
        }
    }
    defer_cache_updates = prefill_cache_updates || (dsv41_default_on("LLAMA_DSV41_DEFER_CACHE_UPDATES") && is_dsv41_decode() &&
        cparams.n_seq_max == 1 && cparams.offload_kqv && model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
        model.n_gpu_layers() > hparams.n_layer_all);

    for (int il = 0; il < n_layer; ++il) {
        if (defer_cache_updates && (is_dsv41_decode() || !cache_updates.empty()) &&
                il > 0 && model.dev_layer(il) != model.dev_layer(il - 1)) {'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        if (il == prefill_kv_only_layer) {
            GGML_ASSERT(n_outputs == 0 && cache_updates.empty());
            return;
        }'''
new='''        if (il == prefill_kv_only_layer) {
            // Truncated prefill still has to populate every remote replica.
            for (const auto & updates : cache_updates) {
                for (auto * write : updates.second) {
                    ggml_build_forward_expand(gf, write);
                }
            }
            cache_updates.clear();
            GGML_ASSERT(n_outputs == 0 && cache_updates.empty());
            return;
        }'''
assert s.count(old)==1;s=s.replace(old,new);f.write_text(s)
(E/'.gitignore').write_text('artifacts/candidate/\nartifacts/*.so\nartifacts/*-harness\nartifacts/metadata\nartifacts/__pycache__/\nartifacts/*/scopes.jsonl\nartifacts/*/*.so\n!*.log\n')
import shutil,subprocess
shutil.copy2('/tmp/implement-prefill-cache-deferral.py',P/'implement.py')
(P/'runtime.patch').write_text(subprocess.check_output(['git','diff','--','src/models/deepseek41.cpp','src/llama-kv-cache-dsv4.cpp','src/llama-kv-cache-dsv4.h'],cwd=R,text=True))
print('Implemented guarded opt-in, unchanged decode conditions, KV-only flush.')
