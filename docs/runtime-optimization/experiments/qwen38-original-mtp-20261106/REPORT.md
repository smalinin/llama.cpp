# Original Qwen3.8 MTP execution, 2026-11-06

Status: completed measurements. The original source and binary are unpatched. A private copy of the nonshared MTP GGUF changes one metadata integer; inference tensor bytes are unchanged. Our runtime source/defaults and the user's configuration are unchanged by this experiment. Measurements and binary identities were captured before committing the preceding prefill fix and these reports together.

## Compatibility and identity

The supplied nonshared GGUF makes both the user's original b12138 and the freshly built local original e14a779719652200cec7ad58dfa4820edc347182 abort before readiness with exit-6. Both traces go through common_context_can_seq_rm, kpool::set_input, set_input_k_idxs and a null buffer. These are failed startup attempts, not speed measurements. Core dumps were disabled for the test subprocesses.

The ordinary MTP GGUF's 49-entry qwen4exp.attention.compress_ratios array ends in0 despite exporting the MTP indexer. A separate copy sets only element48 to4, matching our existing compatibility inference. The changed integer starts at byte2012; all other bytes compare equal. The file is4137429120 bytes, tensor data starts at10946688, and the complete unchanged tensor-data SHA256 is e3020a01fc4f934127210f63e67ed1a06afb14a22ccea4ada286081ee378ac7c. Source size/mtime and the original zero value were checked again after all runs. The source GGUF was not modified.

Temporary copy: /tmp/qwen38-original-compat/mtp-Qwen3.8-Flash-Next-Q8_0-ratio4.gguf. The archive contains its manifest and the recreating/verifying helper. The shared GGUF was not adapted for original loading; the original still requires its own embedding/head.

Original e14 was built in /tmp/glm53-reference-cuda from the unchanged /home/sergei/Github/orig_llama/llama.cpp tree, Release CUDA SM86/89. Server/tools were enabled in the existing build directory, with UI provisioning disabled for the headless test. Code and kernel sources were not changed. Original git status remains clean. The user's b12138 distribution was also not changed.

A suspected shared-KV detection issue was investigated and rejected: the original context constructor clears ctx_other for Qwen, so the generic MTP test sees false and performs prompt catch-up normally. No compatibility library/interposer or code patch was created. The preliminary user-facing hypothesis was corrected before benchmarking. A copy helper initially assumed UINT32 instead of the actual INT32 array; it failed before copying or modifying a file, then was corrected. Both helper versions are retained.

## Frozen protocol

Same frozen documentation prompt21000, direct /completion, 256 generated tokens, temperature1/top_p0.95/top_k20/min_p0.05/seed1234, ignore_eos=true and return_tokens=true. Every response confirms cache_n0, prompt_n21000 and256 output tokens. These are not the user's exact prompt contents. Each arm has one cold request and three measured warm requests; table values are medians, not peak speed. Stochastic completions differ across arms/repeats, so these are native speed/functionality measurements, not an exact-output or quality test. Server TG uses255 generation steps divided by predicted_ms.

Context262144, B2048/U512, P1, threads20, FA ON, unified F16 KV, mmap/lazy AUTO, fit target512 MiB, cache-ram32768 and indexer budget4096. Main model: the user's six-shard Qwen3.8-Flash-Next-Q8_0 GGUF. Mmproj-F16 is loaded, image-min-tokens1024; all requests are text only. Same six Ada/Ampere GPUs as the preceding MTP experiment. Original MTP and both full-draft runs of our binary use the exact same metadata-adjusted nonshared GGUF. The shared control uses the user's original shared GGUF and our existing ratio inference.

Maximum draft2 throughout. Original has fixed drafting. Our default has LLAMA_MTP_ADAPTIVE unset; the ablation sets only LLAMA_MTP_ADAPTIVE=0. Backend draft sampling remains at its default ON. No evaluation callback or profiler is used. Full argv, selected environment, payloads, output IDs, timings and actual draft counters are saved. Every process checks free GPUs before loading; the watcher stops its own process below512 MiB free. Minimum sampled free memory across attempts is8390 MiB, with no memory-floor stops.

## Results

| Arm | PP median [min,max] | TG median [min,max] | Accepted / drafted | Request wall median seconds |
| --- | --- | --- | --- | ---: |
| Original e14, no MTP | 4522.275 [4517.139,4531.295] | 70.032 [69.784,70.157] | n/a | 8.298 |
| Original e14, ordinary MTP, metadata copy | 3243.627 [3235.807,3272.660] | 72.524 [71.617,82.045] | 440/649,67.797% | 9.992 |
| Our build, same ordinary MTP, adaptive default | 2866.814 [2767.105,2879.405] | 62.580 [58.957,63.827] | 404/553,73.056% | 11.599 |
| Our build, same ordinary MTP, adaptive OFF | 2875.770 [2862.777,2883.622] | 69.257 [54.842,70.671] | 432/661,65.356% | 10.999 |
| Our build, original shared MTP, adaptive OFF | 2878.588 [2761.340,2880.663] | 62.865 [59.816,63.973] | 422/681,61.968% | 11.573 |

Original MTP increases median TG by3.558% while decreasing PP by28.274%; the complete21K+256 request is still slower. The compatibility-adjusted original is faster than our corresponding full-draft runs. The residual contribution of kernel/graph/runtime differences has not been isolated here.

Our default shortens drafts from per-position acceptance history; the original keeps the configured maximum. Disabling our adaptive policy raises the full-draft median from62.580 to69.257 in this series, but one full-draft repeat falls to54.842. Do not infer a guaranteed gain or attribute the full gap to that mechanism. On the original shared file, the OFF control reaches62.865 versus the earlier default shared56.674 (previous experiment); its observed range is59.816-63.973 versus53.549-58.123. That shared comparison is between two serial three-repeat series, not a simultaneous or broad quality evaluation. No default/configuration promotion was made.

The previous full/default and shared/default measurements use different draft variants and generate different continuations. Their differences do not prove tensor-weight differences or prove that the full draft is universally better. Full and shared embedding/head equivalence was not established. The exact-same-file original/our comparison is the relevant controlled comparison in this report.

## Verification, limits and evidence

All successful arms complete four fresh prompt/generation requests and exit0. Failed stock attempts retain their exit-6 traces and controller errors. The same original binary that failed with the source metadata succeeds with the private metadata-adjusted copy. Functional real-model MTP is established for this bounded text workload. Formal PPL/task/distribution checks, real media, longer contexts and a stock unadjusted-GGUF speed result are not established. Shared loading in original remains unsupported by the inspected loader.

Our branch remains my_glm53_flash_1 at c83afd9d3 plus the prior two-file prefill patch; no runtime source change was made for these comparisons. Existing CPU/Ada/Ampere state/export and architecture gates are documented in the earlier MTP report. They are not a numerical gate for the original's complete long-context MTP implementation.

Archive: /home/sergei/_my_old/_my_glm53_flash/qwen38-original-mtp-20261106. It contains all startup failures and successful responses, scripts including corrected helpers, metadata-copy manifest, fixed IDs, summaries/VRAM/exits, build logs/configuration, source identities, both runtimes' binary snapshots and SHA256 manifest. The private model copy is referenced in place and can be recreated from the helper; the original model weights are not duplicated in the archive. Test servers were shut down after each arm; the final GPU query is recorded separately.
