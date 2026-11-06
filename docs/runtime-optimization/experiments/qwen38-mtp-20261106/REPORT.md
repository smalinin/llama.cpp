# Qwen3.8 MTP prefill and upstream load failures, 2026-11-06

Status: implemented and measured on `my_glm53_flash_1`, base `c83afd9d32c226770d6532ba59bac5b158815f61`. Measurements and source/binary identities were captured from the working tree before this commit. CUDA server and existing architecture tests were rebuilt in `build-aliceai`.

## Problem and changes

The user reported matched text-only server results: without MTP PP3442/TG57.5, with MTP PP1600/TG56. These are user measurements, not the helper's fixed documentation text.

Two independent barriers serialized MTP prompt processing. `store_embeddings_nextn_backend()` synchronized the producer before each microbatch copy. The copy now uses the same backend's ordered asynchronous copy API; unsupported backends retain its blocking fallback. The context owns the destination, the public getter synchronizes before consumption, and destruction still synchronizes before releasing buffers. Later producer work is queued after the copy, preserving graph-buffer reuse ordering.

A separate recurrent rollback graph mismatch invalidated the worst-case allocation plan. Long batches had 7377 layout nodes, short decode had 7449 because the initial-state snapshot copy existed only when tokens < K. After decode, every growing PP graph reserved memory again. The snapshot source/destination views and copy now always exist; long batches use zero sequences and a valid slot-zero offset. Short batches still preserve the input state exactly tokens positions back. This changes allocation topology, not GDN arithmetic or rollback semantics. The final target layout is 7485 nodes across PP/TG/PP/TG.

## Frozen protocol

The existing 21000-token documentation corpus and main six-shard Qwen3.8-Flash-Next Q8_0 model are unchanged. Shared draft: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 32 tensors, `qwen4exp.nextn_shared_target_tensors=true`, no embedding or output weights. The trunk has 48 layers; the draft stores layer48. Both contexts use the existing shared-weight and legacy-compression compatibility paths.

Context262144, B2048/U512, P1, FA ON, unified F16 KV, threads20, mmap/lazy AUTO, cache-ram32768 MiB, fit target512 MiB. MTP type draft-mtp, maximum draft2, backend sampling ON. Target placement remains 9/9/9/9/9/3 on the five 48 GiB Ada cards and one 24 GiB RTX3090; draft layer on CUDA5. Weights, sampling defaults and kernel selectors are unchanged.

The helper clears both memories and prefills the fixed IDs, then replays eight teacher-forced tokens through target and MTP catch-up to force the graph transition before the next request. It synchronizes after each logical 2048-token batch in all arms, separates target and catch-up times, and includes catch-up in PP speed. It does not generate speculative drafts or measure acceptance/TG. This differs from the earlier native port helper, which synchronized only at the end of plain prefill; do not compare its 4499.916 directly with this control.

There are two cycles per arm, one cold and one warm, not a three-repeat median. Native timings exclude Nsight and evaluation callbacks; scheduler timing counters are enabled. Every GPU launch checks that no existing compute process owns a GPU, and the watcher stops its own process below 512 MiB free. A busy-GPU attempt exited20 before loading; no user process was stopped. Minimum sampled free memory is 9120 MiB for MTP and 13572 MiB for plain control.

## Results

| Variant | Cold PP | Warm PP | Warm target / catch-up seconds |
| --- | ---: | ---: | --- |
| Baseline MTP | 1586.539 | 1712.881 | 10.997947 / 1.262100 |
| Async copy only | 2464.317 | 1754.492 | 10.712840 / 1.256438 |
| Async copy and stable rollback graph | 2461.911 | 2803.962 | 6.238192 / 1.251210 |
| Plain control, final build | 3112.824 | 3498.111 | 6.003239 / 0 |

The combined fix improves warm PP by 63.70% in this series. Async copy alone does not solve repeated requests: after decode, 42 PP reserves drain the backend for 6.677221 seconds. The final two-cycle target trace records 86 allocations, one node count, zero reallocation/replan, and zero reserve synchronization. Necessary synchronization at the logical target/draft boundary remains. MTP PP is still about19.84% below plain control, principally its 1.25-second catch-up; target time itself is about3.91% longer.

A separate Nsight diagnostic captured only the first MTP prefill of the async-only variant. GDN AR totals0.404240 seconds versus0.256265 MMA seconds in the earlier plain profile; this does not explain the large regression. No MMA eligibility change was made. Kernel sums are not critical-path wall time and profiler results are not native throughput.

## Correctness and limits

CPU/Ada/Ampere temporary export checks pass for masked and unmasked outputs spanning four microbatches, append, state restore, rollback, clear/reuse and destruction with queued work. Hidden-output and state checks require NMSE <1e-6; CPU export is exact. Existing Qwen and GLM architecture checks pass on all six GPUs and CPU references, including synthetic MTP/shared-head and rejection/replay cases.

The existing state helper passes full/per-sequence restore, partial restore plus trim, sequence copy, truncated-restore recovery and repeated-position M-RoPE splitting on CPU/Ada/Ampere. Its CPU trace changes from two node counts with eight reallocations to one node count with zero reallocations. Actual real-model shared MTP catch-up completes both cycles, with finite full-vocabulary PP logits and equal top1. Long-model CUDA outputs are not bit-identical across baseline repeats: baseline repeat max/RMS1.047289/0.203727; candidate repeat0.628578/0.110366; warm candidate versus baseline1.748198/0.361991. These are diagnostic screens, not a formal PPL/task quality gate.

At the prefill-only stage, real speculative generation/acceptance/TG was not run; the follow-up below now measures it. Nonshared draft execution is covered by the later original-MTP comparison. Image/video requests, long-context admission, and other backends/devices remain unmeasured. Synthetic M-RoPE checks do not establish real multimodal integration.

## Real server PP/TG follow-up

After the user freed the GPUs, the same final server and libraries were exercised through `/completion` on a private loopback port. No further runtime source or configuration change was made. The mmproj was loaded with image-min-tokens1024, but requests contain only the same frozen 21000 text token IDs. Context262144, B2048/U512, P1, unified F16 KV, FA ON, threads20, mmap/lazy AUTO, fit512 and indexer4096 are explicit in the saved argv. Source/binary hashes are unchanged from the prefill fix.

Each arm runs four fresh prompts with cache_prompt=false: one cold request and three warm measurements. Every response confirms cache_n=0, prompt_n=21000 and256 generated tokens. Sampling is temperature1/top_p0.95/top_k20/min_p0.05/seed1234, ignore_eos=true, return_tokens=true; recorded generation settings confirm the requested values. The server's TG definition is255 generation steps divided by predicted_ms. Model text and stochastic completions differ from the user's actual prompt and are not a distribution/quality test.

Order: plain, shared MTP maximum2, shared MTP maximum1, plain repeated. MTP uses default backend draft sampling, p_min0 and n_min0. The response reports active draft-mtp and nonzero generated/accepted draft counts. Rates are warm medians with ranges in tokens/s; acceptance is the aggregate over the three warm requests.

| Arm | PP median [min,max] | TG median [min,max] | Accepted / drafted | Median request wall seconds |
| --- | --- | --- | --- | ---: |
| Plain first | 4274.905 [4269.027,4281.355] | 59.577 [59.079,59.645] | n/a | 9.209 |
| MTP maximum2 | 2849.299 [2845.688,2850.246] | 56.674 [53.549,58.123] | 375/560,66.964% | 11.882 |
| MTP maximum1 | 2903.735 [2877.103,2919.988] | 59.067 [55.382,59.953] | 329/434,75.806% | 11.626 |
| Plain repeated | 4249.588 [4229.479,4252.561] | 58.883 [58.858,59.747] | n/a | 9.282 |

This corpus establishes functional real shared-MTP generation, rollback and repeated full prefill through the server, including mmproj initialization. It does not establish an MTP speed gain. Maximum1 is roughly equal to plain TG within the observed variation; maximum2 is slower. PP remains about32-33% below the real server plain control, so complete 21K+256-token requests are slower with either MTP setting. The extra draft/verification work does not pay for itself in this series. No default or user configuration is changed, and no result for maximum3, a different prompt or other model is inferred.

The first plain controller sent SIGINT once. This fork requires a second Ctrl+C within two seconds, so the controller eventually killed its own idle server after all four successful responses (exit-9). Its request timings are retained with that cleanup limitation. An attempted MTP admission while that own process still existed exited20 before loading. The controller was corrected to SIGTERM; both MTP arms and the repeated plain control exit0. All four measured response sets are complete; the repeated plain arm supplies a clean-exit control. Minimum sampled free VRAM is9062 MiB, no memory-floor stops occurred, and the final host GPU query is empty. These controller failures did not interrupt user processes or modify runtime code.

Evidence is under the existing archive's `server-tg/`: exact runner versions, expanded argv and selected runtime environment, input payloads, complete responses including output IDs, timing/counter summaries, logs, VRAM and exits, and measured identities. `REPORT-prefill-only.md` retains the earlier report. No Nsight or evaluation callback was used for these server measurements. Formal PPL/task quality and real media integration remain untested.

## Upstream failures reported by the user

The crashing server uses `/home/sergei/_llama_cpp/_llama_def/llama.cpp-b12138`, distinct from the local Git original at `e14a779719652200cec7ad58dfa4820edc347182`. Relevant model/cache files differ mainly in graph API/RoPE refactoring; both paths were inspected. Do not label the distribution source as that Git commit.

1. The shared GGUF intentionally omits `token_embd.weight` and `output.weight`. The inspected upstream Qwen loader requires a private embedding unconditionally, so both the fit estimate and actual draft load fail. Our existing loader permits omission and borrows weights from the target context.
2. The nonshared GGUF contains both tensors and loads, then crashes in `common_context_can_seq_rm -> process_ubatch -> llm_graph_input_kpool::set_input -> set_input_k_idxs -> ggml_backend_buffer_is_host -> ggml_backend_buffer_get_type`. Its 49-entry compression array ends in zero for MTP layer48, despite that layer exporting indexer tensors. Upstream registers k-pool inputs while QSA is bypassed for that layer; `k_idxs` is unused and has no allocated buffer, but the setter still writes it. Our existing compatibility code infers ratio4 from the consistent trunk when the MTP indexer exists, and creates the k-pool input only for an active QSA path. These compatibility changes predate this performance patch.

No original source, original binary, model file or user server configuration was changed. The supplied stack is enough to identify the null-buffer path; we did not reproduce the original crash under a debugger in this experiment.

## Evidence

Raw archive: `/home/sergei/_my_old/_my_glm53_flash/qwen38-mtp-20261106`. It retains baseline and async-only binaries, final sources/binaries and patch, helpers and fixed IDs, metrics, PP logits, native logs/argv/environment/VRAM/exit records, Nsight report/SQLite, check logs, source identity comparisons and SHA256 manifest. The original timing arms and failed busy-GPU attempt remain evidence, not successful final runs. Final server: `/home/sergei/Github/llama.cpp/build-aliceai/bin/llama-server`.

## Original execution and fixed-draft follow-up

The [next experiment](../qwen38-original-mtp-20261106/REPORT.md) runs the unpatched original e14 binary after adjusting only the last compression-ratio metadata integer in a private ordinary MTP GGUF copy. Original plain PP4522/TG70.03, MTP PP3244/TG72.52; the exact same full draft in our build gives PP2867/TG62.58 with default adaptation, PP2876/TG69.26 with adaptation disabled. The latter has a wide54.84-70.67 range. Our original shared file with LLAMA_MTP_ADAPTIVE=0 gives PP2879/TG62.87. These bounded results extend the earlier default-policy test; no defaults were changed.
