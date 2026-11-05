# DeepSeek41 verification widths 1..6, 2026-11-05

Completed read-only diagnostic after runtime commit 3854c3785. The sixth query costs materially more than the fifth: 8192: +25.52 ms / +59.30%; 90112: +22.14 ms / +48.12%. The observed branch expands MoE dispatch and strict floating per-query matmuls; extra graph construction/instantiation does not explain the width 6 cliff. No runtime source/default/preset/deployment changed in this diagnostic.

## Conditions and timing definition

Same new-host Threadripper PRO 3945WX / 12 threads / 192 GB/recorded six-GPU UUID order, Q3_K-AntirezLike target with MXFP8 Engram, Q8_0 KV, TopK 2048, context 92160, B8192/U512/P1, all 15360 experts on GPU with frozen overrides, expert cache=0/staging=0. Same BF16-named mixed-weight DSpark model loaded and feature/tail registration active; K5 and n_rs_seq=5 retained for every width. New prefill deferral explicitly 0; CPU lookup early unset/default ON. Existing private libllama from the preceding validated prototype is loaded in both arms, with unchanged production GGML modules. Full argv/env/source/model/binary identities: [run manifest](artifacts/results.json), [build](artifacts/build.json).

At prefix 8192 and 90112, use the same six fixed input token IDs truncated to each width1..6, without sampling/proposal selection. Before each target call restore identical sequence state and synchronize, outside the timer. Time llama_decode plus llama_synchronize; collect logits/hashes/state afterward. Prefill and feature processing are outside these intervals. Backend sampling is disabled so all requested logits are available. This includes raw-logit output handling and is not the native server GPU-sampling path, total TG or a DSpark acceptance benchmark.

Two fresh processes: clean target-call timing with no interposer, then a counter-only process. The latter increments counters while target decode+sync is armed; it adds no eval callback, CUPTI, backtrace, per-API timestamps, numeric reads or per-call JSON. A separate memory monitor enforces 512 MiB sampled free on every GPU. Two passes traverse widths in opposite order, each with five measured repetitions; first pass has two warmups, second one. All 156 observations per arm are retained,312 total. Counters are diagnostic; timing below comes only from the uninstrumented process.

## Uninstrumented target call medians

| Queries (anchor included) | Prefix8K, ms | Prefix88K, ms |
| --- | ---: | ---: |
| 1 | 24.233 | 24.333 |
| 2 | 30.181 | 31.729 |
| 3 | 34.831 | 36.395 |
| 4 | 38.981 | 42.240 |
| 5 | 43.032 | 46.021 |
| 6 | 68.551 | 68.166 |

Each cell uses ten measured calls. Ranges, pass medians and counter overhead are in [summary](artifacts/summary.json). These medians are not an estimate of total server TG: no draft forward, proposal selection, feature bridge, acceptance/rejection decision or complete request is measured here. The six proposed IDs are fixed inputs, not fresh DSpark proposals.

## Operation and dispatch counts

| Prefix | Queries | Backend graph submissions | GGML nodes | Floating MUL_MAT | MUL_MAT_ID | CUDA Graph launch API | Event wait API |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 5 | 44 | 9245 | 275 | 120 | 41 | 291 |
| 8192 | 6 | 114 | 17423 | 950 | 720 | 111 | 465 |
| 90112 | 5 | 43 | 7115 | 275 | 120 | 40 | 256 |
| 90112 | 6 | 113 | 14860 | 950 | 720 | 110 | 408 |

MUL_MAT_ID expands120->720, exactly three expert projections x40layers x6separate query graphs instead of one grouped token batch. Floating matmuls expand 275->950. Quantized ordinary MUL_MAT remains 331 in this cohort. This identifies existing shape/dispatch fragmentation, not proof that any proposed implementation will achieve the full extra-cost reduction. API counts do not measure physical kernels, transfer bytes, wait durations or a unique additive share of runtime.

All measured llama graphs report reused_delta=1 and allocation_graph=0. Width6 has zero CUDA Graph instantiations in every measured counter row. After changing shapes in the reverse pass, one of ten width 5calls instantiates 39 graphs on 8K / 38 on 88K; isolated width1/2/3cold rows also exist. They are preserved, not removed. Medians and both pass medians show the same large six-query cliff; cold rows cannot explain width 6 being slower. An earlier progress message saying no measured instantiations existed was too broad; this paragraph and the raw counters give the actual scope.

## Existing source boundaries

- is_dsv41_verify_batch(kv_only=false) permits2..5queries; its KV-wide branch already permits6. Six queries with n_rs_seq=5 are decode/verification, not prefill.
- build_dsv41_mm uses a4Dsample view for guarded floating batched GEMV through width 5; width 6 falls back to individual strict per-query floating matmuls and concatenation. Its quantized sample path already uses the six-query KV-wide eligibility.
- DeepSeek41 MoE grouping uses the generic five-query predicate; width 6 calls the existing moe function six times. In the frozen tensor layout the first five layers have Q3_K/Q3_K/Q4_K expert weights, the other 35 layers Q3_K/Q3_K/Q3_K.
- GGML_CUDA_MMVF_SAMPLE_BATCH shares weight loads only through five samples. Existing generic six-sample kernels are present, but enabling their shared-load selection is a separate candidate from graph grouping.
- The current Ada MMVQ thresholds admit Q3_K through 6 columns and Q4_K through 7; the current Ampere path admits this cohort. That source fact does not replace six-query numeric/rollback verification or establish Q2/Q4-model, other-GPU or CPU-cache support.

[Exact source regions](artifacts/summary.json) preserve source SHA256 and line-numbered excerpts. The next candidate should isolate grouped MoE6 from floating sample-grid6, without globally changing every caller of the generic max_tokens predicate or enabling F16 compact attention unintentionally.

## Integrity, memory and limits

All same-width repeated full logit vectors match. Target state hashes captured at the last repetition of each pass also match (24 checkpoints per arm), and base-state restore is exact for both prefixes. Native/counter logits and split counts match for all 156 observations; sampled per-width states, base snapshots and fixed IDs match separately. First-position logits are also exact across widths 1..6. This is benchmark integrity of unchanged code, not new-model quality proof or an all-position serial-vs-batch gate.

Both processes exit 0 without forced kill, allocation-size mismatch, CUDA/OOM errors or floor breach. Sampled minimum free: native 1397.125 MiB, counters 1391.125 MiB. Source/modules and target model stat identities are unchanged; no processes remain on GPU after completion. Original code commit remains3854c3785. [Harness](artifacts/width.cpp), [counter source](artifacts/counts.cpp), [controller](artifacts/run.py), [analysis](artifacts/analyze.py), [logs](artifacts/run.log). Initial unexecuted harness/build snapshots are retained; before launch only incremental observation persistence was added to preserve partial evidence on failure.

## Next bounded work

Prepare independent opt-in/defaultOFF graph extensions: first grouped MoE for the sixth query using existing expert kernels, then floating sample-grid6. Retain old5query/other-mode fallbacks, Q8/text/P1/full-GPU scope, expert IDs, weights, precision/reduction order, causal masks, taps, recurrent state and rollback. Any combined candidate needs full logits/taps/KV/state/NLL/rejected-window replay gates and fresh native same-K OFF/ON/OFF before adoption. Shared-weight MMVF6 selection is a later, separately measured kernel step. No default promotion or promised TG gain from this diagnostic.
