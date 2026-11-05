# Independent CPU lookup before the GPU transition fence

Status: guarded default ON on `my_maya` by explicit user request after the exact gates below. Unset/1 enables `GGML_SCHED_CPU_LOOKUP_EARLY`, 0 disables. The original native and precision measurements used the private opt-in build; their source/library hashes and checks remain unchanged. See [default promotion](#default-on-promotion) for separate build and default checks.

The previous new-host comparison and telemetry were committed as `aa33d0490`. This experiment applies the [user-approved design](../../designs/1017-cpu-lookup-overlap.md) to that base. All comparisons below use the new motherboard/CPU/RAM and explicit current GPU UUID order. Old-platform performance is historical evidence only.

## What changed and why it is safe within its guards

Only `ggml/src/ggml-backend.cpp` changes. The scheduler can start an independent CPU GET_ROWS split before waiting for the preceding CUDA split. This hides CPU lookup/host dispatch behind existing GPU work. It introduces no worker, stream, extra buffer, arithmetic change, or new graph topology.

Early dispatch requires prepared asynchronous inputs, protected copy slots, no evaluation callback, one real CPU backend, and discrete CUDA backends. CPU computation must consist of GET_ROWS from immutable host weight buffers using owned host input row IDs, with read-only views only. The proof rejects CPU expert math, GPU-derived IDs, mutable weight storage, exported CPU outputs, direct GPU access to host compute storage, and staged MoE. Each lookup result must be exported through a layout-matching INPUT/OUTPUT-protected host stage separate from CPU scratch and then a device destination copy.

The existing asynchronous-input API already rejects shared backend buffer types, which could otherwise alias gallocr scratch across streams. The prototype reuses that guard. Current and preceding graphs must both pass isolation checks. The first eligible graph after an unsafe or failed predecessor keeps the old ordering. Slot retirement, input preparation, protected staging, H2D/D2D waits, GPU dependencies and allocator fallback fences remain in place.

This matters because all three CPU lookup groups can reuse the same scratch address. Removing every zero-input CPU fence without checking the staging/consumer contract would be unsafe. The implementation removes only the proven redundant previous-GPU host fence.

## Frozen configuration

Q3_K-AntirezLike, ten SSD shards including MXFP8 Engram, same model files and input token IDs as the [new-host comparison](../1017-maya-newhost-20261105/REPORT.md). Top-K 2048; context 100352; batch 8192; ubatch 512; parallel 1; 12 CPU threads; Q8_0 target K/V; all 15360 target expert bundles on GPU with the same fixed GPU-only tensor overrides. Expert cache and prefill staging are OFF, fitting is OFF, no mmproj. Pipeline and its GPU-override guard are enabled.

DSpark is the existing BF16-named file, whose actual payload is predominantly MXFP4. Native timing uses maximum 5 drafts, p-min 0.5, full policy and the same two-GPU draft placement. Precision uses p-min 0 and raw-logit backend sampling OFF; it is a separate correctness gate, not a timing comparison.

Full argv, environment, library identity, UUID order and responses are in [native-results.json](artifacts/native-results.json). Production base SHA256 remains `511d69ebf26eb47d1ea41045040c3219acbb98342cc44a3ffcc6741298fe1b68`. The private candidate is loaded with LD_LIBRARY_PATH; `/proc/PID/maps` confirms it in every native arm.

## Native counters before the change

An external counters-only probe measured the current production binary. No CUPTI, backtrace sampling, evaluation callback, CUDA API interposition or runtime source change. The probe records the scheduler's event-sync callsite and CPU graph boundaries; it does not create a complete GPU critical-path trace.

| Prefix | Wide ubatches | Previous-GPU transition wait | CPU lookup groups | Total CPU lookup time | Slot retirement/preparation waits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 16 | 32 calls, 2.9998 s | 48 | 0.2097 s | 0.000689 s |
| 98304 | 192 | 384 calls, 40.9987 s | 576 | 2.6231 s | 0.008480 s |

The production-library callsite `0x3fd2a` maps to the zero-input transition fence. `0x3ba26` and `0x46595` are the smaller preparation/allocation slot waits. The counter scope covers wide target chunks, not the final four-token output chunk, draft work, or every runtime operation.

Inclusive host waiting contains necessary GPU computation. It is not all removable time. The candidate overlaps the much smaller CPU lookup/dispatch work; it does not eliminate 3 or 41 seconds of required GPU work.

The probe's PP8K pre/counts/post was 1338.82 / 1348.93 / 1348.38 t/s; PP98K was 1380.79 / 1377.97 / 1378.69. All nine 16-token responses matched per-prefix IDs and exited cleanly. These controls make the lightweight counters credible for this branch; they do not validate the earlier heavy-profiler native TG overhead. See [counter records](artifacts/counts.jsonl), [controller and responses](artifacts/counts-results.json), and [derived summary](artifacts/summary.json).

## Clean native OFF / ON / OFF

Three fresh processes use the same private library, changing only the startup flag 0 / 1 / 0. No interposer, CUPTI, evaluation callback or timing diagnostics. Each prefix has a discarded warmup; 512 and 8K have two measured requests per arm, 98K one. Each request performs the full uncached prefix and produces 128 greedy tokens; TG uses 127 decode intervals.

| Prefix | PP OFF-pre | PP ON | PP OFF-post | PP gain vs mean OFF | TG mean OFF | TG ON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 719.37 | 726.80 | 730.18 | +0.28% | 37.25 | 38.35 |
| 8192 | 1336.98 | 1425.44 | 1338.31 | +6.56% | 43.94 | 44.83 |
| 98304 | 1382.02 | 1454.66 | 1379.84 | +5.34% | 46.38 | 47.15 |

Values are tokens/second; paired repetitions are averaged within each arm. At 8K the individual ON PP results are 1424.18 and 1426.71, compared with OFF 1334.77-1341.84. At 98K there is only one measured request per arm. The experiment supports a bounded wide-prefill gain. Small TG differences and OFF drift do not establish a general TG improvement. PP512 is unchanged within variation.

All 24 responses, including warmups, match output token IDs and generated/accepted draft counters exactly. Native ON records 1808 early transitions on the target scheduler; contexts without eligible CPU work record zero. All processes exit 0, with no floor breach or CUDA/OOM/allocation-mismatch warning. The sampled minimum free VRAM is identical across arms: 1373.25 MiB during loading and 1271.25 MiB during inference, above the required 512 MiB. Sampling is not an instantaneous allocation trace.

## Scheduler correctness and fallback checks

The final [external harness](artifacts/scheduler.cpp) has 14 variants, each tested with explicit OFF, ON and unset: 42 successful cases. It uses real CUDA work before CPU GET_ROWS and a final GPU consumer. Every output element matches the reference and all three modes exactly. The positive 48-chunk case observes 47 early CPU starts while the preceding GPU event is not yet complete. The first graph stays conservative.

Checks cover view exports, widths 1/4/6/7/128/511/512/513, more than four copy slots, graph rebuilds and allocation growth, an unsafe predecessor, mutable weights, exported CPU output, GPU-backed row IDs, extra CPU math, callbacks, unprepared inputs, single-copy mode, controlled compute failure/retry, and two shared-buffer-type layouts including a graph without a GPU data edge. Unsafe modes retain old ordering. Failure/unsafe predecessors cannot enable an early start in the following graph.

The native screen originally followed the 36-case harness. The late shared-buffer audit added six cases; all 42 were rerun against the exact same candidate before final precision. [Unit manifest](artifacts/unit-manifest.json) records source/library hashes. Staged MoE and direct host GPU access are conservative code exclusions, not claims of full-model performance coverage. This is not an all-model or GPU sanitizer certification.

## Full model precision and lifecycle

Two fresh processes, explicit OFF and ON, exited 0. All 12 paired result groups (24 records total) match exactly on two 98K corpora, technical prose and C++. The comparison covers raw target/draft logits, taps 37-39, sequence/global KV state, first-8K state, full/sequence/partial restore, 90K resume, K3/K5 drafting, rejection/partial/full rollback, and append widths 1/7/511/513. All target-state and next-draft replay checks are true.

| Corpus | Teacher-forced positions | Mean NLL OFF | Mean NLL ON | Delta |
| --- | ---: | ---: | ---: | ---: |
| Technical prose | 128 | 0.6571422382475399 | 0.6571422382475399 | 0 |
| C++ source | 128 | 0.30007071394157364 | 0.30007071394157364 | 0 |

These NLL rows are an exact transport-regression check, not a broad model quality benchmark. A scheduling-only change must match exactly; no numerical tolerance is used.

Each process injects two controlled seed-compute failures, one per corpus, then verifies both contexts cleared and exact first-8K state after retry: four recoveries passed. Error lines occur only inside the marked FAULT_BEGIN/FAULT_END spans. Existing fused-op fallback and tap-producer peer-fallback warnings are retained; no CUDA/OOM/compute-buffer mismatch occurred. This does not simulate a GPU fault or real OOM.

The precision sampled minimum free VRAM during inference was 1325.12 MiB OFF and 1327.12 MiB ON; both remain above 512 MiB. ON records 1182 early transitions. [Per-arm results](artifacts/precision-1/results.json), [controller](artifacts/precision.py) and [harness](artifacts/precision.cpp) preserve all checks and corpus fingerprints. Raw model log bytes are retained unchanged; log review decodes invalid UTF-8 with replacement only in memory.

## Provenance and preserved failed attempts

[Build metadata](artifacts/final-build.json) identifies source SHA256 `934640b999ae81c183768e028ee819231de052ff9382f93f4d0e8f70ff86607e` and candidate SHA256 `3a4db1449292d1445c7a881ec484aa453ac95802c25a5de03b286b7355e53d91`. These match the measured native build, final scheduler harness and running precision. Model weights, production binaries and CUDA kernels were not modified. `build.py` builds only a private scheduler library using existing CMake flags/objects; it must not run concurrently with a test loading that file.

A temporary duplicate shared-buffer guard was removed after confirming the existing async API already rejects the layout. Its patch/build manifest is retained, but that abandoned `candidate-final` library is not the final candidate and must not be used for reproduction. [Audit record](artifacts/shared-buffer-audit.json) confirms byte-identical final source/library to the native measured build.

The first shared-type negative fixture incorrectly required the async API to succeed; it failed before graph execution. That fixture and log are preserved as `shared-fixture-initial-*`; the corrected fixture expects rejection and passed. This was a test expectation error, not a failed runtime compute gate.

An initial precision OFF loader was deliberately terminated with SIGTERM before READY/prefill while auditing shared-buffer protection. Its original exit -15 manifest/log, controller and failed supervisor queue remain as `initial-precision-*`, `initial-queue-*`, and [cancellation record](artifacts/precision-cancellation.json). It was not a GPU crash. The final precision uses fresh OFF/ON processes; cancelled evidence is not counted as passed.

Final [review checks](artifacts/review-checks.json) confirm source/library identity, unchanged production hashes, target-shard metadata and the complete sidecar hash. [Reproduction notes](REPRO.md) describe the single-use controllers and final fixture admission. No compute processes remain on the GPUs.

## Use and next step

The initial prototype was OFF until review. The user subsequently requested default ON; in the promoted build an eligible pipeline enables the path without an extra lookup flag:

```bash
LLAMA_PIPELINE_PREFILL=1 ./llama-server ...
```

The existing GPU-only override eligibility remains required when overrides are present (`LLAMA_PIPELINE_GPU_OVERRIDES=1`); CPU spill/staged MoE does not become eligible. `GGML_SCHED_CPU_LOOKUP_EARLY=0` retains the old ordering. The original private validation did not replace production libraries; the subsequent promotion build is recorded separately below.

This narrow fix does not close the full Maya prefill gap. Next, measure the residual prefill dependency/copy path with the candidate enabled, distinguishing actual overlap from GPU polling and different upper-layer work. The six-query target-verification float/MoE path is a separate next experiment. Q2/Q4, other models/backends, media, parallel >1, expert-cache/staging paths, actual OOM/cancellation recovery and larger-context fitting remain separate gates.

## Default ON promotion

After the exact gates and review above, the user requested default ON and a commit on 2026-11-05. The only runtime difference from the fully measured prototype is the constructor expression: unset or `1` enables the existing guarded path, `0` disables it. Other non-`1` explicit values also disable it. No isolation guard, arithmetic, copy slot or dependency changed. [Default-only source delta](artifacts/promotion/default-only.patch).

The adapted external fixture passed all 42 cases against the private default-ON library, then all 42 against the installed production module. In both runs unset and explicit ON enable eligible overlap, outputs remain exact to OFF, and explicit zero/unsafe modes retain original ordering. See [private checks](artifacts/promotion/results.json) and [installed checks](artifacts/promotion/production-results.json). These are dispatch/default checks; the native PP numbers above remain the original explicit-ON measurements. The full model precision suite was not rerun for a constructor-default-only change.

`cmake --build build-aliceai --target ggml-base -j 12` completed successfully. The existing llama-server loads the updated library and `--version` exits 0, confirmed by the loader trace. The installed scheduler object is byte-identical to the tested private object. The full library hashes differ because CMake also regenerated `GGML_COMMIT=aa33d0490-dirty` and rebuilt ggml.c; [separate hashes and checks](artifacts/promotion/production-library.json) preserve that distinction. All other recorded runtime library hashes are unchanged.

The initial full llama-server target began rebuilding unchanged CUDA templates and was intentionally stopped as unnecessary for this scheduler-only change. Its original log and [cancellation record](artifacts/promotion/production-build-cancel.json) are retained. This was not a compile failure or GPU fault. Only the required scheduler module was installed; no model/preset or external runtime directory was changed.

Current usage needs no extra lookup flag in an eligible pipeline. To disable it before starting the parent server/router:

```bash
GGML_SCHED_CPU_LOOKUP_EARLY=0 ./llama-server ...
```

The current source, default, knowledge base, original evidence and promotion checks are committed together on my_maya by user request. The earlier OFF/private-only statements in raw manifests describe their historical test stage. All existing unsupported-mode exclusions and next-step scope remain unchanged.
