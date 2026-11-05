# DeepSeek41 bounded MoE5+1 for six-query verification, 2026-11-05

Status: operator gate passed; full-model exactness is not established. The initial rollback mismatch and a later pre-candidate prefill mismatch require reproducibility isolation. Experimental selector remains OFF; no native speed gate or adoption. LLAMA_DSV41_VERIFY_MOE_WIDE=1 selects5+1groups inside the existing view/concat MoE loop for six-query/K5 text verification. The earlier direct6 operator gate failed on Ampere Q3_K because its MUL_MAT_ID MMVQ cap is5; that negative candidate and source/binary/probe evidence remain in ../1017-verify-moe6-20261105. No CUDA kernel or dispatch threshold is changed here.

Scope guards: exactly6queries/n_rs_seq5, P1/text/causal flash/layer split/full declared offload/pipeline, no eval callback/warmup, Q8 CSA/LID storage, contiguous F32 activation and contiguous CUDA router/separate Q3_K/Q4_K expert tensors. Other cases retain old single-query/five-query paths. This reuses existing tensor views, concat and grouped MoE semantics, without scheduler/event/copy-slot/attention/tap/draft-policy changes. Floating batching remains capped at5globally; no Phase B/C changes or promotion/deployment authorized in this stage.

Canonical implementation is runtime.patch and final source snapshot; scripts/counter/module hashes will bind every gate. Tests compare to same-day frozen OFF references with unchanged hardware/placement/inputs and exact arithmetic contract. Actual memory>=512MiB and clean native same-K OFF/ON/OFF remain required before adoption.

## Current gates

Private libllama SHA256 `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3`; production modules unchanged. Fresh no_alloc admission at ctx92160 retained all15360 GPU expert bundles and gives a conservative minimum1265.30859375MiB after recorded overhead/reserve allowances. This is an estimate; actual monitoring remains required. The 5+1 operator fixture passed all64 cases on Ada/Ampere Q3_K/Q4_K, including actual up/down dimensions and repeated/rotating expert IDs, bitwise exactly against scalar dispatch.

The full-model raw-logit controller stopped after the first corpus rollback group differed from the frozen same-day OFF reference. Its SIGTERM exit(-15) is the controller's deliberate abort after comparison failure, not an observed GPU crash. Prefix, K3/K5 draft and full/partial/sequence restores matched. All recorded rollback logits and next-draft equality assertions passed, but the target serialized state differs for six-query keep3 and target_replay is false for keep6. This fails the chosen exact state contract. No NLL/fault-completion or native speed result is claimed.

The failed [precision manifest](artifacts/precision-results.json), partial [raw result](artifacts/precision-1/results.json) and [differences](artifacts/initial-precision-differences.json) remain intact. A fresh OFF/ON diagnostic partitions the actual state serializer into metadata/tensor spans over the already-captured host bytes, retaining the original GPU state-read count and rollback ordering. Its purpose is to locate the difference and check the OFF reference, not to relax the gate or measure speed. Runtime source is frozen while it runs.

## Fresh controls and repeat: initial failure remains unresolved

The [state-component diagnostic](artifacts/rollback-probe-analysis.json) completed fresh OFF and ON processes with exit0. Four first-corpus groups match each other and the archived OFF reference. All six rollback variants have identical before/after serialized bytes and identical128-part metadata/tensor descriptors across arms. No metadata changed during descriptor construction. Actual sampled minima: OFF1453.125MiB, ON1455.125MiB. This diagnostic adds CPU serializer/hashing work and uses a separate executable without the fault interposer; it does not invalidate the initial failure or establish timing-independent correctness.

The [repeat of the original harness](artifacts/precision-repeat-results.json) retained the original executable, fault interposer, inputs and expanded model arguments; only output paths differ. It stopped earlier: [34 recorded fields](artifacts/repeat-differences.json) differ in prefix/draft groups, including taps and target/draft state at the first8K prefix and last prefix logits. The new six-query MoE path cannot activate there: target ubatches are512queries and the final output call is4queries. [Identity checks](artifacts/repeat-identity-check.json) confirm unchanged source/module hashes and model arguments; the environment difference is the trace output path. The controller deliberately terminated this run(-15), with no floor breach. No GPU crash or quality-loss magnitude is inferred from these hashes.

The separate [fresh OFF prefix control](artifacts/baseline-prefix-results.json), using the same fault interposer without arming faults, processes the first8K three times with existing clear/reset operations. All three state/tap groups match each other and the canonical archived first8K reference; exit0. Actual sampled minimum: 1441.125 MiB. This validates these three observed controls, not universal baseline determinism. The initial rollback failure, instrumented pass, earlier prefill mismatch and passing OFF repeats are all retained; their shared cause is not established.

## Decision and next isolation

Do not promote the selector or claim a speed/NLL/failure-retry result. The full12-group/two-corpus check was not completed. Production libraries/presets are unchanged; the guarded source prototype is saved with these diagnostics and remains OFF for review, with its measured binary/source snapshots preserved. Direct6 remains rejected by its independent, reproducible Ampere Q3_K primitive mismatch.

Before further batching or speed work, isolate verification from independent prefill variation: compare OFF/ON starting from the same fully captured target/draft prefix state and frozen input IDs, preserving context/graph reset and startup-selector semantics. Capture raw changed state spans at the first failure without inserting extra work before the failing operation, and separate prefill reproducibility, verification arithmetic and graph/slot lifecycle checks. A passing perturbed diagnostic must not replace the failed gate. Phase B floating sample-grid6 and Phase C weight-sharing kernels remain unstarted.

## Subsequent shared-prefix isolation

[Completed OFF/ON/OFF diagnostic](../1017-verify-shared-prefix-20261105/REPORT.md), after commit e69c07128: all117 target cases and18 standalone K3/K5 checks match from the same captured90112-token target/draft prefix. All three waves/rollback replays are exact. Actual six-query MoE nodes720->240->720 and backend submissions113->57->113; floating nodes950 unchanged. Minimum free1443.125MiB, exits0. This isolates the checked verification mechanism from independent prefill values; original full-prefill/rollback failure attribution remains unresolved. No native speed, NLL completion or default promotion. Next: first8K raw forensic capture before changing runtime.

## Subsequent raw controls and full conditional repeat

[Nine fresh/reset OFF first8K captures](../1017-prefill-forensic-20261105/REPORT.md) match exactly. The [full two-corpus conditional-capture repeat](../1017-full-precision-forensic-20261105/REPORT.md) passes12groups, all restores/rollback/append transitions, original NLL4delta0 and two seed failure/retries; raw capture count0, minimum1387.125MiB, exit0. Original uninstrumented failures remain unresolved; the diagnostic layout differs and no exactness requirement is relaxed. No new default/runtime or speed result. Next: same-prefix cost and native OFF/ON/OFF with matching controls.


## Subsequent controlled performance gate

[Fresh target-width and native OFF/ON/OFF report](../1017-verify-moe5plus1-performance-20261105/REPORT.md) now provides speed evidence for this same private library: six-query target latency -20.26% at8K/-15.84% at88K, native speed table with matching27output arrays/draft counters and sampled memory>=512MiB. This updates the historical speed-gate status above without erasing any failure or claiming their cause fixed. Default remains OFF; source and production modules unchanged. Next is the independent floating sample-grid6 candidate.
