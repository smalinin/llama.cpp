# DeepSeek41 prefill cache deferral, 2026-11-05

Status: approved prototype implemented and validated in the bounded configuration below. Default OFF. It improves measured PP8K by 4.45% and PP88K by 7.40%, but regresses PP512 by 4.74% and uses more joint target/draft/tap workspace on CUDA5. It remains opt-in for review. No new commit, default promotion or production deployment was performed in this implementation stage.

## What changed

The graph reuses existing remote CSA/LID cache_updates and SET_ROWS operations. Eligible text prefill ubatches defer remote replica writes until the first local reader. KV-only early return explicitly expands every remaining write and clears the map before the existing n_outputs=0/empty-map assertion. Local and unrecognized-owner writes stay eager. Existing decode selection/ordering is retained.

Initial scope: DeepSeek41 token input,2..512 tokens per ubatch, one sequence/slot, causal flash attention, layer split, full GPU attention/FFN/routed/shared weights, pipeline enabled, Q8_0 CSA/LID storage and no eval callback. CPU Engram tables remain supported. Embedding/media input, parallel slots, unsupported cache types/backends and partial offload use the original prefill path. Source graph guards are separate from the narrower measured cohort below; they do not establish general model support.

No new arithmetic, quantization, input/output pointers, scheduler events, protected copy slots or cross-graph pending jobs. A nonvirtual forwarding cache type accessor avoids adding graph nodes just to inspect storage. The private module was built from two changed CXX objects and existing unchanged build objects; class layouts did not change.

[Approved design](../../designs/1017-prefill-cache-deferral.md), [preceding owner audit](../1017-kv-owner-audit-20261105/REPORT.md), [exact runtime patch](artifacts/runtime.patch), [build commands and identities](artifacts/build.json). The initial implement.py edit trace predates guard refinements; runtime.patch and final source snapshots define the built code.

## Frozen cohort

New host: Threadripper PRO 3945WX, 12 CPU threads, 192 GB RAM and the recorded GPU UUID/PCI order. Source baseline f796e544f, documentation HEAD cf1e5a461. Fixed Q3_K-AntirezLike target with MXFP8 Engram, Q8_0 K/V, TopK 2048, B8192/U512/P1, all 15360 routed expert bundles on GPU, fixed tensor overrides, expert cache=0 and prefill staging disabled. No mmproj/auto-fit. CPU lookup early remains unset/default ON for every arm.

DSpark: same BF16-named file as the baseline, actual mixed weights (mostly MXFP4), K5/full/pmin=0.5, draft CUDA5/CUDA0 and the same layer1 down override. The filename is not a claim that all draft tensors are BF16. Exact argv/environment, model stat identities and library/source hashes are recorded in [native manifest](artifacts/native-results.json). Models and presets were not modified.

Context 100352 ON failed conservative 512 MiB admission at 464.23 MiB on CUDA5. Contexts 99328/98304/94208 also failed. Per the user's prior authorization to reduce context, both OFF and ON use 92160. Long prefix 90112 is 88K, not 98K. The chosen conservative bound is 519.86 MiB after previously measured per-device overhead plus 64 MiB guard. No speed comparison to earlier ctx=100352 cohorts or old-platform measurements is made.

## Clean native speed

Three fresh processes OFF/ON/OFF using the same private module. Every prefix has one warmup plus two measured repetitions in each process. The table compares ON's two measured repetitions with the four measured OFF repetitions bracketing it; warmups are excluded. Each request has a full uncached prompt and 256 greedy output tokens. PP uses prefix length/prompt time; TG uses 255 intervals/predicted time. No interposer, profiler, eval callback or diagnostic observer in this speed cohort. The external memory monitor is identical in all arms.

| Prefix tokens | OFF PP t/s | ON PP t/s | PP change | OFF TG t/s | ON TG t/s | TG change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 724.48 | 690.17 | -4.74% | 41.57 | 41.65 | +0.20% |
| 8192 | 1412.14 | 1474.92 | +4.45% | 45.42 | 45.41 | -0.02% |
| 90112 | 1449.66 | 1556.94 | +7.40% | 60.88 | 61.80 | +1.52% |

| Prefix tokens | OFF measured PP range | ON measured PP range | OFF-post vs OFF-pre mean |
| --- | ---: | ---: | ---: |
| 512 | 716.12..731.13 | 689.78..690.55 | -0.51% |
| 8192 | 1403.41..1422.95 | 1472.37..1477.47 | -0.69% |
| 90112 | 1447.37..1451.49 | 1554.59..1559.28 | +0.08% |

The medium/long PP gains exceed the measured control drift, while the PP512 regression also repeats across controls. Stable TG acceleration is not established: its ON repetition spread overlaps or approaches the controls; this is a short screen, not a statistical confidence interval or a long-generation quality comparison. No fresh matched Maya run at this smaller context was performed.

All 27 response token arrays and draft generated/accepted counters match exactly for each prefix, including warmups. Every prompt reports cache_n=0 and the full requested prompt length; each response contains 256 tokens. All three processes exit 0 without forced termination, allocation-size mismatch, OOM, CUDA errors or 512 MiB sampled-floor breach. [Summary with per-arm means/ranges and memory](artifacts/native-summary.json), [controller](artifacts/native.py), [run log](artifacts/native-run.log).

## Precision and lifecycle

Actual flags 0/1 used the same private module and context. Twelve paired result groups (six per corpus; 24 records across both arms) match exactly on two 88K prose/code corpora: full target/draft logits, exported taps, compressor/ring/KV/state hashes, full/partial/sequence restore and replay, K3/K5 drafts and six rejected-window rollback variants per corpus. Logical appended widths 1/7/508/511/512/513 include the 513 split into 512+1 and copy-slot reuse.

Teacher-forced NLL uses 128 tokens per corpus: prose 0.37295498784483416, code 0.1052687732276846, delta 0. Two seed-fault/retries per arm (four total) pass; each arm records 780 calls, 176 injection graphs and two injected failures. Backend sampling is disabled for raw logits; there is no eval callback, only the existing small fault interposer. These exactness checks are appropriate because scheduling changes, not arithmetic.

Actual precision sampled minimum free VRAM: OFF 1387.125 MiB, ON 801.25 MiB, both exit 0 without warnings. These raw checks are separate from native backend sampling/actual allocation above. [Precision manifest](artifacts/precision-results.json), [harness](artifacts/precision.cpp), [controller](artifacts/precision.py).

All 42 existing scheduler forwarding/protected-slot/CPU-lookup/failure cases and the metadata-owner smoke passed after native timing. It validates the unchanged ggml-base module, not the DeepSeek model ordering itself; full-model precision and ownership checks supply that evidence. Results and fixture provenance are in artifacts/scheduler-results.json and artifacts/scheduler-rerun.json.

## Actual ownership and graph structure

Ten paired no_alloc graph variants pass symbolic operator and SET_ROWS operand/count checks, including full-tail and KV-only 511/512 shapes. Node counts remain equal; ON labels seven deferred writes. Constants, named operands and value expressions remain in the fingerprint. Structural checks do not replace numeric/state evidence.

Actual scheduled metadata confirms source20 transformed values now originate on CUDA2 (OFF:CUDA4), without a separate transform-placement patch. All 36 source20 CSA writes remain across nine captured target chunks, with four remote cache allocations and writes before their first readers. Four selected KV-only chunks flush all pending remote replicas. Executed scheduler graphs change 31->24 for KV-only and 50->40 for full-tail chunks; these counts are not a PP speedup ratio.

An additional comparison covers all 419 raw/CSA/LID SET_ROWS operations per arm across the same nine chunks (four with 36 writes, five with 55). Root types, sizes, destination backends, value/index shapes and counts match; every distinct persistent allocation is written once per selected ubatch, before any first actual reader present in that graph. Raw ring, CSA and LID reuse cache_k_lN labels, so root names alone are not allocation identity. The analyzer uses actual root data/storage size within a ubatch, not pointer equality across processes. All ten diagnostic 16-token outputs and draft counters match OFF/ON.

The observer only forwards existing graph/copy/event operations and reads metadata; no numeric device data, new GPU events or CUDA API hooks. Its armed PP timings are perturbed and excluded from the native speed table. Existing protected copy/event paths remain intact. [Actual ON proof](artifacts/owner-on/owner-checks.json), [all cache writes](artifacts/all-cache-writes-checks.json), [same-context default capture](artifacts/owner-default/summary.json). Compressed raw traces have byte/hash round-trip checks in each owner folder's payload.json.

## Memory cost and context limit

Weights, model context and routed placement are unchanged between arms at context 92160. The metadata budget reports CUDA5 joint allocation +795.02 MiB: main compute -4 MiB, draft/taps context +160.02 MiB and draft/taps compute +639 MiB. CUDA4 falls 164.02 MiB, CUDA1/2 each 4.25 MiB; aggregate increase 622.5 MiB. These are existing allocator budget categories, not newly hand-managed buffers or proof of an isolated root cause. Cache write deferral changes live ranges/placement in the joint graph. A workspace reduction would be a separate candidate.

| Native arm | Sampled minimum free during load, MiB | Sampled minimum free during inference, MiB |
| --- | ---: | ---: |
| off-pre | 1431.25 | 1329.25 |
| on | 637.25 | 535.25 |
| off-post | 1431.25 | 1329.25 |

ON actual owner trace minimum 587.25 MiB; native long-prefix inference reaches 535.25 MiB, only 23.25 MiB above the 512 MiB requirement. Metadata estimates alone would miss this distinction. Current admission is for this fixed placement/context and does not prove 262K joint auto-fit, F16 KV, media, parallel>1, CPU expert caching/staging or another model.

## Diagnostic corrections retained

- The first reserve observer was oververbose and deliberately stopped. Its initial source/log/raw trace locator/hash remain; the bounded observer records small headers and at most 12 unique full graph shapes. The 11 GB raw file is ignored locally, not proposed as a versioned payload.
- The symbolic checker initially treated CPY's destination self-link as a value cycle and anonymous FILL shape scaffolding as different values. Only destination/shape-only generated labels were normalized; named operands/constants/value expressions remain. Initial diagnostics and the corrected checker are retained. Runtime source did not change for these corrections.
- The first intended ON ownership controller omitted the flag due to a generator insertion error. Activation checks caught actual default OFF before native timing began. That manifest remains byte-identical under owner-default; owner-default-recovery.json records the mistake. Precision explicitly carried 0/1 and is unaffected. The corrected controller explicitly inserts/asserts 1; actual deferred labels and physical producers prove activation. No runtime source/binary changed to fix the controller.
- The all-cache analyzer initially keyed by cache_k_lN and rejected distinct raw/CSA/LID allocations sharing that label. Actual widths 128/512, storage sizes and root data resolve the identity; initial/debug/final scripts and cache-root-label-check-notes.json are preserved. This was a checker error, not a duplicate runtime write.

- The first scheduler fixture rerun omitted the required metadata-probe environment and stopped at its probe-export check before executing cases. The original JSON/log are retained; restoring LD_PRELOAD and the owner-smoke output path yielded all 42 passes. No runtime, test-body or native timing data changed; see artifacts/scheduler-controller-correction.json.

## Current decision and follow-up

Keep the prototype opt-in/default OFF for review: the PP8K/88K benefit is useful, but PP512 regresses and CUDA5 memory pressure is materially higher. No scheduler fence removal or default promotion follows from these results. A separate candidate could reduce the joint target/draft/tap workspace and investigate the short-prompt regression before widening deployment. The previously planned six-query float/MoE verification work remains independent.

After building this source, LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1 enables the guarded mode at context creation; unset/0 retains eager prefill. Treat it as a startup setting. Existing LLAMA_DSV41_DEFER_CACHE_UPDATES retains its decode default/conditions. Production modules and user presets still use their previous implementation.

Private libllama SHA256: 940a21a2dff9e4419c664b5fcaf111251d5e3071dcecc72317727582dbe45d98.
Production libllama SHA256 remains d2e20751e03c73366e2f5d59e1ebab478b77e45b36b2d6829c8fbd8f39ebfedd.
Production ggml-base SHA256 remains 760e032ea9a7131dd3242ac2036ac8a12c8e7b05882d1dc40c39bebc9ec624bd.
