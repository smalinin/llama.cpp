# DeepSeek41 six-query verification: separate MoE and floating batching

Status: Phase A prototypes tested at operator level; full-model 5+1 rollback exactness is under investigation. The completed read-only [width screen](../experiments/1017-verify-width-screen-20261105/REPORT.md) establishes the baseline on source3854c3785. The startup-only experimental selector remains OFF; CUDA arithmetic and production defaults/modules are unchanged.

## Evidence and scope

At the same prefix and fixed six input tokens, widths1..6 retain n_rs_seq5. Native raw-logit target medians at8K are24.233/30.181/34.831/38.981/43.032/68.551ms; at88K24.333/31.729/36.395/42.240/46.021/68.166ms (use the raw summary for unrounded values). Six versus five costs+59.30%/+48.12%. These calls exclude snapshots/bridge/draft and differ from the server GPU sampler; do not turn their ratio into a promised TG gain.

MoE MUL_MAT_ID count120->720 and floating MUL_MAT275->950; ordinary quantized MUL_MAT stays331. Scheduler graph submissions44->114 at8K,43->113 at88K. All measured llama graphs are reused. Six-query measured CUDA Graph instantiate calls are0. Isolated cold rows exist for smaller widths after shape switches and are retained; medians in both passes still show the cliff.

Initial candidate cohort remains the new-host six Ampere/Ada cards, Q3_K target with Q4_K down experts in layers0..4, Q8 KV, text/P1, full GPU expert placement, context92160 and the same DSpark model/K5/full/pmin0.5. Keep prefill cache deferral OFF while measuring this independent mechanism. Other GPUs, model quants, CPU expert cache/staging/media/parallel are separate validation, not implied support.

## Phase A: grouped MoE at query width6

Reuse the existing grouped build_moe_ffn path at exactly6queries behind an explicit opt-in/defaultOFF selector. Preserve the current2..5predicate and all other fallbacks. Keep precision, router transforms, expert IDs/weights, per-token expert reduction, current quantized kernels, cache writes, masks, taps and recurrent-state handling unchanged. Require the proven CUDA expert/router placement and supported storage; do not route CPU spills into this path automatically.

Do not globally raise is_dsv41_verify_batch's five-query cap: it also controls floating and non-Q8 compact paths. Select the sixth-query MoE case locally. Ordinary quantized sample batching/KV-wide6 already exists and must not be altered for this experiment. First check Q3_K/Q3_K/Q4_K and all-Q3_K expert combinations on the actual Ada/Ampere kernel dispatch; generic backend support alone is not proof of identical reductions.

## Phase B: floating sample-grid at query width6

As a separate flag/candidate, reuse build_dsv41_mm's existing4Dsample view with one GEMV column per query. Preserve contiguous/type/stride/channel/output/precision guards and strict fallback, including grouped wo_a cases that the current optimized path rejects. Initially use the existing generic six-sample MMVF grid; do not combine the graph change with new weight-reuse selection or a CUDA reduction rewrite.

Measure this alone and combined with the proven MoE candidate so shared gains are not counted twice. The current MMVF shared-weight selector stops at5samples; extending that selector is Phase C, with separate operator exactness/performance evidence. Existing switch/template6 code does not by itself establish all shape/stride/precision equivalence.

## Gates for each candidate

- Fixed input IDs/layout/KV/sampler/DSpark policy and startup-only flag selection. Confirm actual grouped dispatch/float path and counter deltas separately from uninstrumented timing; preserve failed or cold rows.
- Operator checks for relevant dtype/strides/IDs and all six positions. Require exact current-reference logits, taps, KV/compressor/ring/source state, full/partial/sequence restore and rejected-window rollback. Include widths1/3/4/5/6, prefix/copy-slot transitions, K3/K5 drafting after verification, failure/reset/retry and both text/code long prefixes. The arithmetic contract is exact, not merely matching greedy outputs or applying the generic0.01NLL tolerance.
- Fresh no_alloc joint admission and monitored real memory with>=512MiB sampled free on every card. Keep expert placement/context identical; if context must fall, establish new reference baselines. Do not infer larger auto-fit coverage from the prior fixed placement.
- Repeat the same-prefix1..6cost/counter screen, then clean same-K OFF/ON/OFF full-server PP512/8K/88K/TG256. Compare K3/K4/K5 as separate policy cohorts; do not require different K policies to share a continuation or credit policy changes to a kernel.
- No default promotion/deployment or merging a faster but numerically different path without review. If a candidate changes no useful native cost or fails exactness, preserve the evidence and leave OFF or revert its code.

No scheduler events/copy slots/fences, host input ownership, cross-graph asynchronous tasks or different Maya KV/Engram formats are part of these bounded extensions. Such changes need their own ownership/lifetime design.

## Phase A update, 2026-11-05

[Direct6](../experiments/1017-verify-moe6-20261105/REPORT.md) fails exactness on Ampere Q3_K: the actual MUL_MAT_ID selector caps MMVQ at5 there, unlike Ada. The bounded [5+1 follow-up](../experiments/1017-verify-moe5plus1-20261105/REPORT.md) passes64 primitive cases but stops at full-model rollback state comparison. Fresh OFF/ON state-component diagnostics pass, while the original-harness repeat diverges already at first8K before the candidate can activate; three fresh OFF8K reset repeats match the canonical reference. The cause remains unresolved; isolate verification on common saved prefix state before more batching/speed work. Speed gates and Phases B/C have not started. Do not infer full-state equivalence from the passed linear operator/logit assertions.

## Shared-prefix gate, after e69c07128

[OFF/ON/OFF isolation](../experiments/1017-verify-shared-prefix-20261105/REPORT.md) passes117 target cases/18draft checks, exact restored prefix and repeated state/logit/rollback comparisons. MoE720->240 and backend submissions113->57; floating950 unchanged. This is a diagnostic, not a native speed/full acceptance result. Original failures remain unresolved; the next bounded task is raw first8K prefill forensics with no added work before the failure point. Defaults stay OFF; Phase B/C have not started.

## Full conditional repeat after raw first8K controls

[Nine raw OFF groups](../experiments/1017-prefill-forensic-20261105/REPORT.md) and the [complete two-corpus repeat](../experiments/1017-full-precision-forensic-20261105/REPORT.md) now pass the observed exact gates, including two failures/retry. No conditional capture triggered. Old failures are not declared fixed; the diagnostic executable changes host allocation history. Original NLL4 does not exercise the new wide branch, which has separate exact full-logit/state/rollback checks. Next is read-only same-prefix/native cost with strict matching controls, while retaining full-prefill reproducibility as an open promotion item. Default stays OFF; Phase B/C remain unstarted.
