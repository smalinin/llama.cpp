from pathlib import Path
import json,subprocess,shutil,hashlib
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-verify-moe6-20261105';P=E/'artifacts';assert not E.exists();assert not subprocess.check_output(['git','diff','--','src'],cwd=R,text=True);P.mkdir(parents=True);(E/'.gitignore').write_text('artifacts/candidate/\nartifacts/*.so\nartifacts/*-harness\nartifacts/__pycache__/\nartifacts/*/scopes.jsonl\n!*.log\n')
D=R/'docs/runtime-optimization/designs/1017-six-query-moe-float.md';shutil.copy2(D,P/'reviewed-work-plan.md');f=R/'src/models/deepseek41.cpp';shutil.copy2(f,P/'reference-deepseek41.cpp');old='''        const char * batch_moe = std::getenv("LLAMA_DSV41_VERIFY_MOE_BATCH");
        const bool batch_verify_moe = is_dsv41_verify_batch() ||
                (cur->ne[1] <= 3 && batch_moe && std::strcmp(batch_moe, "1") == 0);''';new='''        const char * batch_moe = std::getenv("LLAMA_DSV41_VERIFY_MOE_BATCH");
        const char * wide_moe = std::getenv("LLAMA_DSV41_VERIFY_MOE_WIDE");
        bool wide_verify_moe = wide_moe && std::strcmp(wide_moe, "1") == 0 &&
                n_tokens == 6 && cur->ne[1] == 6 && cur->ne[2] == 1 && cur->ne[3] == 1 &&
                cur->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) && is_dsv41_verify_batch(true) &&
                cparams.type_k == GGML_TYPE_Q8_0 && cparams.type_v == GGML_TYPE_Q8_0 &&
                !cparams.cb_eval && cparams.pipeline_parallel && model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
                model.n_gpu_layers() > hparams.n_layer_all && !layer.ffn_gate_up_exps;
        if (wide_verify_moe) {
            const auto on_cuda = [](const ggml_tensor * tensor) {
                if (!tensor || !tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer) || !ggml_is_contiguous(tensor)) {
                    return false;
                }
                auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
                return dev && std::strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0;
            };
            const auto expert_on_cuda = [&](const ggml_tensor * tensor) {
                return tensor && (tensor->type == GGML_TYPE_Q3_K || tensor->type == GGML_TYPE_Q4_K) && on_cuda(tensor);
            };
            wide_verify_moe = on_cuda(layer.ffn_gate_inp) && expert_on_cuda(layer.ffn_gate_exps) &&
                    expert_on_cuda(layer.ffn_up_exps) && expert_on_cuda(layer.ffn_down_exps);
        }
        const bool batch_verify_moe = is_dsv41_verify_batch() || wide_verify_moe ||
                (cur->ne[1] <= 3 && batch_moe && std::strcmp(batch_moe, "1") == 0);''';s=f.read_text();assert s.count(old)==1;f.write_text(s.replace(old,new));shutil.copy2(f,P/'candidate-deepseek41.cpp');(P/'runtime.patch').write_text(subprocess.check_output(['git','diff','--','src'],cwd=R,text=True));shutil.copy2('/tmp/implement-verify-moe6.py',P/'implement.py')
(E/'REPORT.md').write_text('''# DeepSeek41 opt-in grouped MoE6, 2026-11-05

Status: Phase A implemented, validation pending, default OFF. This is the existing grouped MoE path extended locally to the sixth query; no new scheduler/lifetime/kernel pattern. It does not raise the generic verification predicate or change floating sample selection, quantized kernels, attention/cache writes, masks, taps, draft policy or defaults. New selector LLAMA_DSV41_VERIFY_MOE_WIDE=1 is startup-only and initially limited to text/P1/Q8K+V/full-GPU layer split/pipeline/no eval callback, six all-logit verification queries and contiguous F32 activations, contiguous CUDA router and separate Q3_K/Q4_K expert tensors. Unknown/host/fused/other-type cases retain the old per-query fallback. Other hardware/model cohorts remain unvalidated; exactness is required on the frozen Ampere/Ada layout before adoption.

Baseline diagnostic/design committed3966ac2fd, external plans35c2969. Tests reuse existing external operator/full-model/state/rollback harnesses in isolated modules, with>=512MiB actual free and fixed source/library/model identities. User request is commit and continue through the existing bounded Phase A work; default promotion/deployment/new implementation commit is not part of this stage.
''')
print('IMPLEMENTED',len(new.splitlines())-len(old.splitlines()),'new lines')
