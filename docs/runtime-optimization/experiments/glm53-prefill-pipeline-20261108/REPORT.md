# GLM-5.3-Flash protected prefill inputs

Date: 2026-11-08. Base: `0d31ed180521a4743ee72d38e3750592f9f86964` on `my_glm53_flash_1`.
Status: measured experimental opt-in, committed at user request; default OFF, no installation replacement or default promotion.

## Problem and change

The ordinary GLM target used the synchronous scheduler input path even when general pipeline parallelism was enabled. Graph reuse fences and input/allocator ownership restricted submission across microbatches. The candidate reuses the existing protected input arenas, completion events and no-reallocation planning path for GLM. It changes no model operations, weight types, attention top-k, reduction kernels or graph operator ordering.

`LLAMA_GLM5NEXT_PIPELINE_PREFILL=1` permits this path for a single-sequence DEFAULT GLM context with scalar positions, layer split, CUDA-resident experts, no other context, embedding output, LoRA, NextN/layer exports or evaluation/abort callbacks. The scheduler still checks its CUDA/event/copy compatibility. Unset/0 retains the original path. Changes to abort callbacks or embedding-output mode schedule a synchronized reserve so that the mode is reevaluated. Existing layer-export and NextN setters already request reserve.

The general pipeline must already be eligible. `LLAMA_PIPELINE_GPU_OVERRIDES=1` was fixed in both native arms, allowing only existing verified GPU overrides. This is not permission for CPU expert offload or arbitrary buffer aliases.

DFlash registers layer exports and therefore remains on its current synchronous path. MTP/NextN consumers are also excluded. Media, arbitrary cancellation, true OOM recovery and longer populated contexts are not covered by this experiment. The prototype is not a default recommendation for those workloads.

## Frozen protocol

- Same unchanged GLM UD-Q5_K_XL six shards, CPU mmproj, no drafter (`--spec-type none`).
- Context278528, batch8192, ubatch1024, unified F16 K/V, FA on, P1,20 threads, load-mode none, fit on/target768MiB, cache-ram0, n-cpu-moe0.
- GGUF default attention indexer top-k2048, no override. Sampling top-k40 is a separate parameter.
- Same50000 saved documentation token IDs and512 outputs; temperature0.95, top-p1, min-p0.05, seed1234, ignore EOS, cache_prompt false.
- OFF then ON, three native requests per arm after a separate2048+64 warmup. Startup/fit time excluded. No evaluation callback or CUDA profiler. The observer only records context initialization and model placement.
- Runtime input/export precision unchanged. Actual tensor buffer/type/bytes maps and layer placement match.
- Startup observer confirms pipeline=true/copies4 in both arms, async_inputs=false/true respectively. Branch eligibility is established separately from speed; no physical DMA or quantified GPU concurrency claim is inferred from the flag.
- Watcher:512MiB sampled VRAM floor,4GiB RAM floor, foreign GPU owner rejection; stop only owned server groups. Both servers exit0.

Expanded argv/environment, request payload and binary hashes are in `native/*/launch.json`, `native/payload.json` and `native/identity.json` in the local archive. Hardware UUID/PCI/driver/CPU identity is in `hardware.json`.

## Correctness

| Gate | Result | Evidence |
| --- | --- | --- |
| CPU fallback, two CUDA devices (Ada/Ampere), all six CUDA devices | PASS; candidate active only on multi-GPU, copies4 | check-all-final.log |
| Multi-microbatch prefill, all-row/final-row logits, ragged append, single-token and four-token continuation | Finite and bit-exact OFF/ON logits | check-pipeline.cpp |
| Sequence state, checkpoint restore, partial rollback with n_rs_seq4 | Exact live sequence serialization | check-all-final.log |
| No-snapshot recurrent mode n_rs_seq0, restore/recompute | Exact logits/state on CPU/two/six GPUs | check-rs0.log |
| Invalid token and subsequent valid decode | Rejected without advancing state; recovery matches | both fixtures |
| Abort callback present, layer export, embedding-output mode | Async path disabled; return to plain mode reenables it; outputs/state match | both fixtures |
| Full-model native outputs | All six512-token arrays identical | native/analysis.json |
| Actual weights/placement | Exact full tensor buffer/type/size and layer maps | native/*/placement.jsonl |
| Actual memory | Sampled minimum832MiB OFF,824MiB ON; no guard stops | native/*/memory.jsonl |

Sequence serialization proves live state equivalence, not unused physical cache cells. Full-model raw logits/NLL, true GPU allocation failure and full278528-token population are not claimed. No numerical tolerance was used to excuse an OFF/ON mismatch.

## Native results

Medians of three requests per arm,50000 input /512 output tokens, no draft:

| Metric | OFF | ON | Change |
| --- | ---: | ---: | ---: |
| Prefill tokens/s | 1019.222 | 2170.751 | +112.981% |
| TG tokens/s | 39.212 | 38.790 | -1.075% |
| Prompt milliseconds | 49057.023 | 23033.506 | |
| Whole request seconds | 62.124 | 36.237 | -41.669% |

PP ranges: OFF1015.433-1021.338, ON2130.175-2184.786. TG ranges: OFF39.161-39.249, ON38.546-38.805. The roughly1.1% TG regression is retained, not reported as an acceleration. This is one corpus and OFF/ON order; no reverse-order control was run. Do not combine these plain numbers with prior DFlash or Maya measurements.

All output arrays hash to `2ee5a3f5c75ad68ef7335576132873b2eafd7064ee16b35acdc0c415a40bb38a`. Reused prompt count is zero in all six native requests. Final tokens_cached is occupied cache size, not reuse.

## Decision and reproduction

Retain as default-OFF experimental code. Prefill improves about2.13x with identical model placement and sampled outputs; this establishes a substantial cost in our former input/scheduling path. It does not assign a precise share of Maya's advantage to a particular scheduler submechanism or prove the remaining TG gap solved. Protected staging and existing early CPU-lookup eligibility can both participate in this path.

The next scope is DFlash feature ownership under pipelined target submission, followed by its own logits/state/draft-counter/native gates. Do not enable hidden-state export by merely deleting its guard.

Local archive: `/home/sergei/_my_old/_my_glm53_flash/glm53-prefill-pipeline-20261108`.

- `candidate.patch`: measured source change (source SHA256 `affa1e61f9402f4e3749f7ca247c22db34c0716811e1ef08a46f2d8796174c20`).
- `runtime-baseline/`, `runtime-candidate/`: preserved shared libraries and servers; use each directory as LD_LIBRARY_PATH.
- `protocol.json`, `hardware.json`, `candidate-binaries.json`, `native/identity.json`: identities and frozen settings.
- `check-pipeline.cpp`, `check-pipeline-rs0.cpp`, compile helpers and logs: functional/ownership gates.
- `run-ab.py`, `placement.cpp`, `native/`: native harness, raw responses, placement and memory logs, summary and analysis.

The measured source and library hashes were checked unchanged after all native requests. The historical suffix-preparation experiment remains archived and removed; it was not reapplied here.
