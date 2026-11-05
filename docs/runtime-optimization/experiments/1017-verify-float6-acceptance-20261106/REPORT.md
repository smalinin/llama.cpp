# Floating6 lifecycle and native acceptance, 2026-11-06

Status: combined MoE1 lifecycle/dispatch PASS; clean native acceptance FAIL at8K. Fresh supplemental OFF controls PASS. Diagnostic ON/OFF tracing completed with matching logits and did not reproduce the failure; it cannot replace the failed clean gate. A subsequent clean ON reproduction also matches original OFF, including all 256 tokens and the tenth token at8K. The original failure remains unresolved. Floating6 remains default OFF; no native speed gain is accepted.

After commit c6be3d990, source and production modules remain frozen. Same private3bb55e36 library as the [paired width screen](../1017-verify-float6-20261106/REPORT.md), MoE5+1 explicit1 and prefill deferral0. The [protocol](PROTOCOL.md) freezes Q3/Q8/TopK2048/ctx92160/B8192/U512/P1/threads12/new-host UUIDs/15360GPU bundles. Prior joint metadata admission is reused only after checking unchanged source/module/model identities; every real process is monitored for>=512MiB free per card.

The external harness reuses the original full precision sequence: two88K technical-prose/C++ corpora, first8K controls, K3/K5 drafts, full/partial/sequence restores, checkpoints, four/six-query rollback, append transitions and two injected seed-failure/retries per arm. Keep NLL128 with4-query chunks, then add NLL126 with6-query chunks after the original fault/retry sequence so the new branch is actually exercised.14groups per arm require exact outputs/state/NLL, plus successful target/draft rollback replay. No tolerance relaxation.

The existing logical backend/API counters wrap only the56six-query target decode/logit calls; they stop before feature processing. They add no GPU reads, eval callback or CUPTI, but CPU instrumentation/allocation history can affect timing and lifecycle. Counter values are not GPU kernel durations or a native throughput result. Fresh OFF/ON share this executable; old uninstrumented failures remain preserved and unresolved. [Harness/build](artifacts/build.json), [lifecycle controller](artifacts/run-lifecycle.py), [frozen configuration](artifacts/config.json).

Only after that gate passes, a clean server OFF/ON/OFF uses K5/full/p-min0.5/backend draft sampling, identical greedy256-output requests at512/8K/88K, one warmup+two measured per prefix, full uncached prompts and reset before every request. All output arrays and draft counts, including warmups, must match. Native TG uses255intervals for256tokens, matching the established runner convention; report OFF drift/ranges. No instrumentation in the speed run.

The MoE0-only and other K-policy cohorts remain separate work. No Phase C CUDA selector change, default promotion, deployment or second commit is part of this continuation.


## Completed lifecycle and dispatch gate

All14groups match across two fresh OFF/ON processes on technical prose and C++ source, including first8K controls, target/draft logits/taps/state, full/partial/sequence restore and checkpoint resumes, append widths1/7/508/511/512/513, four-query keep1/2/4 and six-query keep1/3/6 rollback. All rollback target/draft replay booleans are true. Two injected seed failures per arm clear both contexts and recover exactly; they are deliberate test errors, not observed CUDA crashes. Native speed runs do not use this interposer.

| Six-query logical backend count | OFF | ON |
| --- | --- | --- |
| Floating MUL_MAT | 950 | 475 |
| MoE MUL_MAT_ID | 240 | 240 |
| Quantized MUL_MAT | 331 | 331 |
| Backend graph submissions | 57 | 57 |

This holds for all56paired six-query calls. It halves the counted floating nodes; it does not halve total engine cost. Counts describe submitted logical GGML nodes/API calls, not GPU kernel durations or total kernels inside replayed CUDA Graphs. [Detailed counters](artifacts/dispatch-summary.json), [full lifecycle manifest](artifacts/lifecycle-results.json), [summary](artifacts/lifecycle-summary.json).

NLL128/4 and NLL126/6 each have delta0 between flags, separately on both corpora. All128/126teacher-forced row hashes match across flags; the common126rows also match between widths4and6 within each flag. The two NLL means use different token counts, so their absolute difference is not a quality improvement. The added six-query loop exercises the candidate, unlike the old four-query-only NLL gate.

Actual minima: OFF1381.125MiB, ON1387.125MiB, both exit0 without floor breach/forced kill/OOM/CUDA allocation/compute-buffer expectation warnings. This is observed instrumented equivalence, not universal determinism or a fix for historical uninstrumented failures. The canonical fixture is lifecycle.cpp plus its build manifest; initial prepare.py followed by finalize-fixture.py reproduces the added early first8K JSON controls without new GPU reads. No runtime or production module changed during these gates.


## Clean native failure and separate OFF control

The [original native gate](artifacts/native-results.json) stopped on the ON8K warmup after nine completed OFF requests and three matching ON512 requests. The first nine generated tokens match; generated index9 (the tenth token) differs: OFF469 versus ON27344, after `CMake 3.30`. OFF proposed242/accepted161; ON proposed224/accepted152. The uncached8192-token prompt, generation settings, inputs, source and loaded library identities match. The ON server exits0 after the controller deliberately stops the failed gate, with no observed floor breach or CUDA/OOM/buffer expectation warning. The original manifest remains incomplete; no original OFF-post arm ran. Different output trajectories invalidate its speed comparison.

A [separate fresh OFF post-failure control](artifacts/native-control-results.json) completes all nine requests with exact output arrays and draft counts against the original OFF-pre. These eighteen matching OFF responses in two processes establish the observed controls, not universal determinism or the cause of the ON failure. This supplement does not complete or repair the original OFF/ON/OFF gate.

The [native diagnostic](artifacts/native-trace-results.json) interposes existing decode calls and CPU target-logit getters, capturing metadata and already-returned host logits around the first22 predictions at8K. It adds no GPU read, eval callback or synchronization, but binding, logging and CPU allocations can change behavior. Target sampling is CPU/default; draft backend sampling is GPU/default. Diagnostic timings are excluded from speed evidence. Production modules and runtime source remain unchanged after c6be3d990.


## Decision and next isolation

Do not promote FLOAT_WIDE or quote a native throughput gain. The guarded MoE5+1 default remains unchanged; this decision concerns the separate floating6 experiment. The accepted evidence is limited to the primitive/width screen and the combined instrumented lifecycle/counter/NLL gates. A clean native mismatch remains a failed acceptance gate even when a later instrumented run matches.

Before another speed series or CUDA selector change, repeat the original uninstrumented ON process with the same reset/512-to8K history and frozen module identities. If reproducible, replay its captured batch IDs/positions/output rows from a common captured target/draft prefix; separate startup/reservation and prefill state variation from the first actual six-query branch, and compare the first changed logits/state span. Preserve original failing output and distinguish repeatability from root-cause attribution. No speculative synchronization, arithmetic or graph-cache fix is implemented by this report.


## Diagnostic ON/OFF completed

The [diagnostic summary](artifacts/native-diagnostic-summary.json) records two clean exit0 processes and four matching256-output responses per arm:512 warmup/two measured, then8K warmup. All eight match the original OFF arrays and draft counters; the original failed ON8K output is not reproduced. All1365 decode events per arm have identical target/draft token IDs, positions, output rows and request order after removing the monotonically assigned event ID. All22 full host-logit rows at predictions8192..8213 are bitwise identical across ON/OFF, with no nonfinite values. This includes the last prefill output before the first actual six-query verification in the8K request. Observed minimum free1359.25MiB in each arm; no warning or forced kill.

This is diagnostic equivalence only. The interposer explicitly binds libllama, logs metadata and copies already-returned CPU logits; even without new GPU reads/synchronization, it changes binding, CPU allocations and timing. It cannot establish the cause of the clean failure, prove it is harmless, or establish native acceptance. Diagnostic PP/TG numbers are retained in the raw manifest and excluded from performance conclusions. Raw logit buffers stay local/ignored; recorded SHA256/top10/bits, source, runner and comparison script are retained.

[Comparison script](artifacts/analyze-native-diagnostic.py) validates raw hashes, request controls, full decode metadata, bit equality and memory floors; the original failed manifest remains unchanged and incomplete. No experiment server is intentionally left running. New evidence is saved separately from commit c6be3d990.


## Clean ON reproduction after fdf8e354a

At the user's request, a fresh [ordinary ON run](artifacts/native-clean-repeat-results.json) repeats exactly the first four original requests:512 warmup/two measured, then8K warmup, with the same reset before each request. Same private3bb module, runtime source hashes, weights, expanded argv/environment (apart from the server alias), placement and greedy settings. The new commit contains documentation/evidence only. No LD_PRELOAD/interposer, callback, profiler or added target-logit access; process maps verify diagnostic libraries are absent.

| 8K response | Tenth generated token (index9) | Start | Draft accepted/generated |
| --- | --- | --- | --- |
| Original OFF | 469 | `CMake 3.30 or higher` | 161/242 |
| Original failed ON | 27344 | `CMake 3.30+` | 152/224 |
| Fresh clean ON repeat | 469 | `CMake 3.30 or higher` | 161/242 |

All 256 output tokens and draft counters in all four repeated requests match their original OFF counterparts; the8K failure does not reproduce even without instrumentation. This removes any basis for assuming only the diagnostic interposer hides the failure, but does not identify its cause or show that it was fixed. No runtime/default/library change was made to obtain this pass. This one fresh ON process is a reproduction check, not a new accepted OFF/ON/OFF speed series; the original failed acceptance manifest remains unchanged and incomplete.

[Summary and identity checks](artifacts/native-clean-repeat-summary.json), [runner](artifacts/native-clean-repeat.py), [comparison script](artifacts/summarize-native-clean-repeat.py). Sampled free memory stays>=1359.25MiB; server exits0, no observed CUDA/OOM/buffer expectation warning, floor breach or forced kill. The test server is stopped. Keep FLOAT_WIDE OFF pending reproducibility/root-cause work. New repeat evidence is subsequent to commit fdf8e354a.
