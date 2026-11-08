# GLM-5.3-Flash protected prefill inputs with DFlash2, 2026-11-08

Status at measurement: inconclusive experimental candidate; default OFF, uncommitted. Current source default was subsequently changed to ON at explicit user request; see the policy update below. This extends the separate plain-GLM optimization committed as `2ee901564`. It does not promote or replace that experiment's evidence.

## Mechanism and scope

The candidate permits `LLAMA_GLM5NEXT_PIPELINE_PREFILL=1` when every active target layer tap has registered producer-local CUDA storage. Partial or extra taps retain the previous fallback. Bridge binding/removal requests a scheduler reserve, and speculative initialization and fit reserve the target after successful binding. The existing protected scheduler input arenas, four copy slots, events and replan handling are reused. DFlash feature input consumption remains synchronous; no precision, operator, weight, attention top-k, sampler or draft policy change is intended. Default OFF remains unchanged. MTP/NextN, media, multiple sequences and unsupported consumers are outside this experiment.

## Frozen protocol

Base: `2ee9015644d9d04383fa4707c42b0814839b9c87`. Local evidence: `/home/sergei/_my_old/_my_glm53_flash/glm53-dflash-pipeline-20261108`. `candidate.patch`, `protocol.json`, `hardware.json`, `candidate-binaries.json` and `native/identity.json` preserve measured source, hardware and binary identity; `runtime-candidate` preserves the measured libraries. The installed `my_build1` binaries were not replaced.

Target: unsloth `GLM-5.3-Flash-UD-Q5_K_XL`, six shards. Draft: Anbeeld `GLM-5.3-Flash-DFlash2-Q5_K_M.gguf`. CPU mmproj loaded, text-only requests. Same 50,000 saved token IDs from `glm53-mmq-20261106/glm53-prefill-50000.tokens`, followed by 512 generated tokens, no EOS stop. Native warmup: 2,048 IDs plus 64 outputs, then three repeats per arm, OFF followed by ON in fresh processes. Temperature 0.95, top-p 1, sampling top-k 40, min-p 0.05, seed 1234. No prompt reuse (`cache_prompt=false`, `cache_n=0`).

Context 278,528, batch 8,192, ubatch 1,024, unified F16 KV, FA on, P1, 20 threads, fit on/target768, load mode none, all experts on GPU. The target attention indexer top-k is the GGUF default 2,048; this is separate from sampler top-k and the draft's selector/window. DFlash maximum4, minimum0, p-min0, all draft layers on GPU. `LLAMA_DFLASH_GPU_FEATURES=1` and `LLAMA_PIPELINE_GPU_OVERRIDES=1` were fixed in both arms. Only `LLAMA_GLM5NEXT_PIPELINE_PREFILL` changes.

The startup observer records actual buffers and final activation after binding; it does not install an evaluation callback or instrument native compute. A memory watcher stops only the owned process group on a foreign GPU owner, free GPU memory below512MiB or available RAM below4GiB. Expanded argv/environment, raw responses and logs are in each arm's directory.

## Native observations

| Metric | OFF | ON |
| --- | --- | --- |
| Median PP, tokens/s | 937.769 | 1802.459 |
| PP range, tokens/s | 932.686-938.180 | 1755.330-1814.563 |
| Median TG, tokens/s | 61.992 | 60.739 |
| TG range, tokens/s | 61.931-62.137 | 60.601-60.750 |
| Median request wall time, seconds | 61.610 | 36.185 |
| Proposed / accepted draft tokens, every repeat | 479 /391 | 483 /390 |
| Acceptance fraction | 81.628% | 80.745% |
| Minimum sampled free GPU memory, MiB | 1112 | 1106 |

Observed PP improves92.207%, wall time decreases41.268%, and TG decreases2.023%. These rates do not establish a correct or production-ready optimization: native draft counters differ consistently. `native/analysis.json` retains `same_draft_counters=false` and `valid_controlled_comparison=false`.

All six 512-token output arrays have SHA256 `2ee5a3f5c75ad68ef7335576132873b2eafd7064ee16b35acdc0c415a40bb38a`. Actual target and draft tensor maps match, including buffer/type/size for1383 target and81 draft tensors and all layer devices. Both target modes use pipeline=true, four copies and five producer-local taps; only the protected-input activation differs. Matching sampled tokens is not a substitute for matching internal states and draft decisions.

## Correctness and diagnostics

The synthetic GLM/DFlash fixture passes on CPU, two CUDA devices and all six CUDA devices: finite/exact logits, packed versus host taps, target/draft serialized states, selector lattice, append/ragged batches, restore, RS4 rollback, invalid input/retry, bridge removal during pending work, host fallback, rebind and extra-tap fallback. `check-all.log` records each `DFLASH_GATE ... exact PASS`; helper sources and build logs are archived.

A separate busy-input negative control shows that the generic synchronous packed-tensor setter alone does not protect a destination that is still being read. This is not evidence that the production caller violates its ownership contract, and no speculative synchronization fix was applied on this basis.

The full-model short diagnostic (`diagnostics/analysis.json`) runs eight identical2K+64 jobs in one process with complete target/draft physical memory clearing and fresh scheduler reservation before each job. Jobs0-3 do not add draft input waits/captures. Jobs4-7 synchronize draft input consumption and capture raw target features, last prefill logits and both serialized states. All eight jobs have66 proposed/46 accepted. All four captures of each data class are bit-exact across OFF/ON and repeat. Tiny-server controls also pass. This diagnostic does not resolve the native50K counter mismatch and its timings are not native benchmark results.

The long diagnostic (`diagnostics-long/analysis.json`) repeats the original50K+512 workload in one process. All four jobs clear both contexts physically and re-reserve both schedulers. Jobs0/1 are OFF/ON without extra input synchronization or raw capture; jobs2/3 add synchronization and raw capture. All four have479 proposed/391 accepted, and all outputs match the original native token hash. Captured OFF/ON target features (4,096,000,000 bytes), last prefill logits (619,520 bytes), target sequence state (1,139,434,500 bytes) and draft sequence state (42,567,792 bytes) are bit-exact. No numerical tolerance was substituted for exactness.

This localizes the native difference to initialization/execution history or an effect removed by the combined reset/reserve diagnostic; it does not distinguish physical clearing from scheduler re-reservation or diagnose the production root cause. Additional input synchronization is not required to obtain matching reset-run counters. Do not infer a generic copy-race fix from the isolated busy-input test. The historical native mismatch remains unresolved, and all diagnostic timings are excluded from native speed conclusions. Both diagnostic servers and both native servers exit0 with no watcher stops; the final compute-process query is empty.

## Decision at measurement

Keep the DFlash extension default OFF and uncommitted. Preserve the native mismatch; a reset/instrumented success cannot supersede it. No default promotion, deployment, merge or further commit is performed. The next correctness question is which aspect of native initialization/history causes the changed draft decisions; a changed counter by itself neither proves a quality regression nor validates the transport change.


## Requested default policy update, 2026-11-08

The user explicitly requested default ON after the results and the unresolved draft-counter difference were explained. Unset `LLAMA_GLM5NEXT_PIPELINE_PREFILL` now enables the guarded path, `1` explicitly enables it and `0` disables it. This applies to eligible plain GLM contexts and complete producer-local DFlash target taps. All existing backend, topology, callback, embedding, sequence and consumer guards remain. Callback/embedding mode changes and feature bridge binding/removal use the same unset-default policy for scheduler re-reservation. Other model architectures retain their previous defaults.

This is a policy change, not a diagnosis or repair of the native mismatch. The original explicit-flag native measurements and binary hashes above remain historical evidence. No additional speed gain or quality pass is inferred from changing the default. The user subsequently requested committing the extension, the default change and this report. They are included together in the resulting commit; installed `my_build1` binaries were not replaced.

Build and policy-check evidence: `/home/sergei/_my_old/_my_glm53_flash/glm53-default-on-20261108`. The server and existing architecture-test target were rebuilt. The local launch script `run-glm53-dflash-pipeline.sh` now uses that directory's `runtime/llama-server` and leaves both GLM pipeline and DFlash GPU-feature flags unset unless supplied by the caller. The unset-default versus explicit-OFF fixtures pass on CPU, two CUDA devices and all six CUDA devices for plain GLM and DFlash. They verify actual protected-input activation (false on CPU, true with four copies on multi-CUDA), exact finite logits/features, target/draft states, append/restore/rollback, invalid-input recovery, callback/embedding mode fallback and feature-bridge detach/rebind/extra-tap fallback. `checks.json` records both exit codes0; the six summary gates are in `check-plain-default.log` and `check-dflash-default.log`. Build and runtime snapshot hashes match. `git diff --check` passes. Native speed was not remeasured for this default-only policy change.


## Requested GPU-override permission default ON, 2026-11-08

The user found that the GLM protected-input default still required explicitly enabling `LLAMA_PIPELINE_GPU_OVERRIDES` with auto-fit tensor overrides, and subsequently requested making this permission default ON as well. The shared GPU-override exception now accepts an unset variable or a nonzero numeric value; explicit `0` retains the previous override restriction. The eligibility diagnostic log follows the same default. This does not bypass `tensor_overrides_support_pipeline()` or change its real buffer/device and routed-expert checks, or remove the existing multi-device/layer/KQV/async/event conditions and allocation fallback. Other model architectures can use the same guarded exception, while their independent protected-prefill flags retain their existing defaults.

Local evidence: `/home/sergei/_my_old/_my_glm53_flash/glm53-gpu-overrides-default-on-20261108`. The change is a startup policy only; previous explicit-ON native evidence remains historical. The existing DFlash counter question remains unresolved. GPU tests and full-model benchmarks remain deferred under the user's preparation-only instruction; no new GPU performance or correctness gate is claimed. Production installation and the frozen history-diagnostic runtime are unchanged. The user subsequently requested committing this policy change and its documentation together.

`llama-server` rebuilt successfully. Existing plain-GLM and DFlash CPU fixtures both exit0 with finite/exact logits, state, restore/rollback and fallback gates; `CUDA_VISIBLE_DEVICES` was empty, and linkage logs confirm the rebuilt libllama was used. These CPU checks do not establish GPU activation with an unset override flag; GPU validation remains deferred. The local launch script now points to the new runtime snapshot and no longer exports the override flag, so caller-supplied `0` is respected. `git diff --check` passes.
