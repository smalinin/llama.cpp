from pathlib import Path
import json
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105';P=E/'artifacts'
report='''# Residual prefill after CPU lookup overlap, new host 2026-11-05

Status: host/CUDA API attribution, queue4x screen and private copies8 screen completed. No useful PP gain from either queue depth or extra copy slots. A placement-only shared-expert screen is pending below. No runtime source, installed library, defaults, user preset or weights changed in this experiment. No commit/deploy.

## Frozen scope and method

Runtime `my_maya f796e544f7054affa480e33e73ec2cf8816f25d4`; guarded CPU lookup early remains default ON/unset. Production base SHA256 `760e032ea9a7131dd3242ac2036ac8a12c8e7b05882d1dc40c39bebc9ec624bd`. New Threadripper PRO 3945WX / 192GB host and current UUID/PCI order only; historical hardware timings are not comparators.

Q3/MXFP8 Engram/Q8_0 target K/V, TopK2048, ctx100352, B8192/U512/P1, threads12, fixed GPU-only placement with all15360 target expert bundles on GPU, expert cache/staging off, no mmproj. Existing BF16-named mixed-format DSpark, K5/full/pmin0.5, CUDA5/CUDA0 and existing draft-down override. Exact IDs, expanded command/env, model metadata and runtime/library hashes are in each `counts-results.json`; frozen input source is the preceding new-host experiment. Filenames do not establish actual weight types.

External probes forward existing operations, record host scopes/counters, and add no GPU events, CUPTI, evaluation callbacks or arithmetic. Direct scheduler `iface.cpy_tensor_async` is included, not just public copy wrappers. CUDA attribution is restricted to existing GPU graph_compute scopes. Depth0 API sums avoid nested double-counting. Host duration includes backpressure/unfinished GPU work; it is not GPU kernel time, physical DMA bandwidth or SM occupancy.

Controls are unarmed pass-through requests before/after the armed request in the same process with the interposer loaded. They are not independent interposer-free A/B processes. Two 8K warmups precede controls. All requests are uncached and return16 greedy tokens; TG has only15 intervals and cannot establish a sustained TG gain/regression. Screens reject unpromising candidates; they do not grant default promotion or universal quality support.

## 1. Baseline host scopes, 8K and 98K

The peer-copy API is not the leading measured host-time contributor in this cohort. NVML reports unsupported/CNS peer access, but that does not establish physical transport time or prove copies free of GPU cost.

| Nested scope, host elapsed | 8192 tokens | 98304 tokens |
| --- | ---: | ---: |
| Target decode | 4424.58 ms | 60249.79 ms |
| Target ubatches, graph_compute including CPU | 3811.13 ms | 51388.04 ms |
| Target ubatches, direct copy-interface calls | 75.29 ms | 1827.66 ms |
| Target ubatches, graph allocation | 86.84 ms | 1042.79 ms |
| Feature getter, scheduler drain | 1233.12 ms | 7167.16 ms |
| Full feature bridge | 1265.84 ms | 7503.38 ms |

Do not sum nested rows into wall time. The feature getter waits for unfinished target work; its wait is not drafter math cost. The target's GPU API scopes can block internally despite no explicit synchronize at the outer call.

| Prefix | PP pre | PP armed | PP post |
| --- | ---: | ---: | ---: |
| 8192 | 1419.84 | 1425.67 | 1425.59 |
| 98304 | 1451.43 | 1448.71 | 1443.97 |

[Phase1 checks](artifacts/phase1-checks.json): nine requests exact IDs/draft counters per prefix, exit0, no floor breach/warnings, minimum inference free1295.25MiB. [Full scope summary](artifacts/summary.json), [lossless raw scopes](artifacts/scopes.jsonl.gz).

## 2. API coverage corrected before interpreting host time

The initial API probe counted `cudaLaunchKernelExC`, whereas this CUDA13 library actually imports `__cudaLaunchKernel`. Its partial result, including Sgemm179.72ms, remains in [api-attribution](artifacts/api-attribution/summary.json); it cannot attribute the remaining graph time to CPU calculation. The first probe compile failed on an overloaded CUBLAS function pointer and was corrected before inference; initial source/build diagnostics remain preserved.

The corrected probe forwards the actual CUDA13 compiler launch ABI, kernel lookup/push/pop, stream/capture/event/occupancy and relevant driver/CUBLAS APIs with header-matched types. A 42-case scheduler smoke test passed before full-model inference. [Coverage note](artifacts/hook-coverage-note.json), [corrected summary](artifacts/launch-attribution/summary.json), [caller symbolization](artifacts/launch-attribution/symbolized.json).

At 8K it records51068 actual target launch calls occupying3197.80ms of host API time. Sgemm occupies177.89ms/1440 calls and GemmEx20.08ms/200 calls. Lightweight operators also show long enqueue calls. This is compatible with driver backpressure; it does not mean a lightweight kernel itself takes that duration. Routed tensor types are Q3_K and Q4_K, both supported by MMQ; an unsupported floating expert fallback was not indicated by these node records.

Native pre/post PP1436.28/1425.42, armed1419.95: about-0.76% relative to control mean. Actual minfree1323.25MiB, five requests exact, exit0.

## 3. Queue4x falsification

The process-local `CUDA_SCALE_LAUNCH_QUEUES=4x` screen changes only the existing CUDA queue control. Source, production libraries, copy count4, placement, input IDs and protected events remain unchanged. NVIDIA documents this control as scaling pending kernel/copy queues, not as a guaranteed speedup: [official documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues).

| 8K screen | Queue unset, copies4 | Queue4x, copies4 | Queue4x, private copies8 |
| --- | ---: | ---: | ---: |
| Mean native PP, pre/post, t/s | 1430.85 | 1426.98 | 1418.84 |
| Native PP pre / post | 1436.28 / 1425.42 | 1434.73 / 1419.23 | 1415.36 / 1422.33 |
| Armed launch host time, ms | 3197.80 | 143.02 | 1758.62 |
| Armed target launch calls | 51068 | 51068 | 51068 |

Queue4x reduces launch API time about22x but native PP changes-0.27%, within this small screen's variation. Waiting moves to input-slot retirement and final target drain. It does not eliminate the dependency-limited work.

For queue4x/copies4 the armed target graph_compute sum becomes600.82ms including CPU, while allocate_graph becomes2200.12ms and feature getter scheduler drain2272.12ms. These are nested/instrumented observations, not independent time savings. The event-sync caller `libggml-base+0x45355` maps to `ggml_backend_sched_wait_input_copy` at slot reuse, before graph splitting. It is not a reserve growth or failed allocation fallback. Existing async graph allocation already supports replan without realloc; these data do not justify removing lifetime guards.

[Comparison](artifacts/queue-comparison.json), [API summary](artifacts/launch-queue4x/summary.json), [host summary](artifacts/launch-queue4x/host-summary.json). No queue setting adopted.

## 4. Existing copy-slot count4 -> 8, isolated build

Only the already exposed CMake setting `GGML_SCHED_MAX_COPIES` changes in a private ggml-base build. Production stays4 and is never overwritten. Queue4x is fixed in both copies4/copies8 comparator arms. Source audit confines this macro's arrays to the opaque scheduler implementation. [Exact build/link identity](artifacts/copies8/build.json).

The private library was confirmed in process maps and the log reports `asynchronous pipeline inputs = enabled (copies = 8)`. Forty-two scheduler cases passed with an explicit get_n_copies assertion. No_alloc routes/context/target bundles match copies4; metadata admission uses measured baseline overhead plus64MiB per-GPU guard, followed by monitored real inference.

Native PP8K changes-0.57% against copies4/queue4x. Extra slots move waiting again: allocate_graph90.87ms, launch calls1758.62ms, final getter drain2529.79ms in the armed request. More command/input buffering does not establish more useful chunk overlap here.

All five outputs and draft counters match other cohorts; exit0, no CUDA/OOM/buffer mismatch, actual minimum625.25MiB>=512. This consumes additional VRAM without a useful screen gain. No broader logits/state/rollback/NLL gate was run for copies8 and no adoption is proposed.

The controller's final literal assertion looked for `copies=8`; the actual log contains spaces. All requests and clean shutdown had finished before the assertion. The original [manifest](artifacts/copies8/counts-results.json) remains `complete=false`, and its source/log are unchanged. Separate [recovery checks](artifacts/copies8/checks.json) verify activation, loaded library, hashes, all requests, counters, memory and exit. API/host summaries retain that original incomplete flag; it is a reporting assertion failure, not a GPU failure.

## 5. Dependency audit and next bounded question

[Cross-device wait audit](artifacts/cross-device-waits-audit.json) shows input-free transitions16xCUDA1->CUDA0 before real shared-expert MUL_MAT/GLU at layer6, 8xCUDA4->CUDA3 before real layer29 shared-expert work, and16xCUDA3->CUDA4 before SET_ROWS/VIEW. A tensor named `(view)` is not necessarily a view-only operation: `cache_k_l30 (view)` is SET_ROWS and writes KV. None of these observations proves a fence removable.

The probe captured first/last tensor metadata, not all view roots, strides, root-buffer owners, allocator reuse intervals or latest writers. Reused addresses across graphs cannot establish dependency identity. A scheduler reorder would require an approved new design with actual root/producer/consumer/slot ownership and exact state/rollback gates. No such reorder is implemented here.

A smaller next screen can use existing tensor-placement CLI to put shared gate/up weights6 onCUDA1 and29 onCUDA4, together with already placed down projections. It tests whether placement-induced backward edges are significant, without a new scheduler pattern or guard removal. Admission and actual result are recorded separately below. Source/defaults/user presets remain unchanged.

Maya source snapshots show explicit host-mapped transport/flags and per-device chunk workers. This is an architectural difference, not a measured causal split of its speed advantage. No fresh Maya inference was run in this residual stage; compare only to its separately recorded new-host series.

## Evidence and limits

The first five phases contain29 full-prefix uncached requests including warmups; all same-prefix greedy arrays and draft counters match across phases. Production source/library hashes and target shard stat identities remain unchanged. These transport/queue/config screens do not repeat or replace the prior CPU-lookup raw-logit/state/rollback/NLL gates. Short TG16 is not sustained TG evidence.

Each raw JSONL has a lossless deterministic gzip and checksum manifest, with raw copies retained locally and ignored by Git. Ignored binaries/private objects remain local; source/build commands and recovery diagnostics are versionable. No model data is copied into the evidence. Preserved server logs/responses can include full prompt text. Native requests have no cached prompt reuse.

Current branch changes in this stage are experiment/knowledge documentation only. GPUs are released after each controller. Existing default-on CPU lookup remains as previously committed; no new performance flag is promoted by this report.
'''
(E/'REPORT.md').write_text(report)
