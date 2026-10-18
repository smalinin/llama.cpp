# Step-5 CUDA performance validation

Unless stated otherwise, model measurements use the restored TypeSafeAI Q3_K_M checkpoint with 184 NeuroSenko BF16/F32 indexer tensors. They do not measure a newly converted official checkpoint or establish a broad quality score. The implementation and numerical contract are described in [step5.md](step5.md).

## Conditions

- Intel Core Ultra 7 265K, 20 inference and batch threads.
- Five 48 GiB Ada GPUs and one 24 GiB RTX 3090; CUDA 13, compute capabilities 8.9 and 8.6.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=2,1,0,5,4,3` for every GPU test.
- Context 144000, allocated as 144128; batch 2048, microbatch 512, one server slot, F16 K/V, Flash Attention enabled, context shifting disabled.
- Eager CPU weight loading with `--load-mode none`; CPU expert offload and `--no-op-offload`; separate vision projector on CPU.
- Auto-fit with 2048 MiB device margin generated a placement; paired launches then fixed that placement with fitting disabled. The new indexed attention generated exactly the same fit plan as the earlier direct attention kernel.
- `-ngl 96`, tensor split `17,14,14,14,13,24`, plus the saved per-tensor CPU expert overrides. Target-only and MTP main-model buffers matched; MTP adds its three draft heads and small SWA cache.
- Greedy sampling, seed 1, full uncached prefill (`cache_prompt=false`), warmup with a 7382-token retrieval prompt. Retrieval-31K was run twice per comparison profile. Other long cases were single measurements.
- Fixed MTP comparisons use `--spec-type draft-mtp --spec-draft-n-max 3` and `LLAMA_MTP_ADAPTIVE=0`. Adaptive MTP is measured separately.

Complete local artifacts include requests, responses and output token IDs, server logs, sampled resources, exact arguments, environment, build options, binary SHA-256 hashes and source patches. Old binaries and all their shared libraries were isolated in a snapshot. Repeated launches do not run concurrently, and no other full model was loaded during measurements.

## Indexed attention

The former CSA attention kernel directly computed each selected query/head dot product and softmax with FP32 reductions. It was the dominant prefill operation. The new path maps each provider's logical token selection to physical cache rows and uses the existing indexed tensor-core Flash Attention machinery with 192-dimensional heads. Four providers become four independent attention streams with 16 query heads each. Negative row IDs represent causal or padded entries. Every nonnegative row ID must be below the physical cache capacity; the CSA slot map supplies those valid physical rows.

No full dense mask or duplicated K/V cache is built. Original K/V rows are read using the selected physical indices. Multi-token MTP verification uses the same path. Unusual shapes and separate physical sequence streams retain the direct kernel fallback. Independent logical sequences in one physical cache stream are supported by the row map.

The native CUDA implementation uses F16 probability and accumulator tiles. Its `GGML_PREC_F32` setting must not be interpreted as a promise of FP32 tensor-core accumulation. CPU indexed attention honors the explicit FP32 accumulator request. Floating-point comparisons account for this difference; index selection itself must agree exactly.

For isolated 512-query / 32768-history / 4104-selected-token operator checks, including summaries, scores, top-k, row mapping and attention:

| Device | Direct chain, median ms | Indexed chain, median ms |
| --- | --- | --- |
| Logical CUDA0, Ada | 174.325 | 2.878 |
| Logical CUDA5, Ampere | 422.432 | 7.598 |

These are five measured repetitions after three warmups on synthetic inputs. They are not full-model speedups. On Ada, the profiled direct attention took about 173.53 ms. The indexed attention took about 2.214 ms; top-k 0.228 ms, score processing 0.220 ms, score GEMM 0.145 ms, summary compression 0.053 ms, row mapping 0.051 ms and copies 0.056 ms. Profiling and timing runs are separate.

## Full-model measurements

| Mode | PP, tokens/s, two runs | TG, tokens/s, two runs | Output tokens per run | Stop |
| --- | --- | --- | --- | --- |
| CSA direct, fixed MTP 3 | 63.71 / 63.64 | 12.66 / 12.46 | 93/93 | eos/eos |
| Legacy dense, fixed MTP 3 | 149.84 / 149.65 | 17.60 / 18.15 | 94/94 | eos/eos |
| Corrected dense, fixed MTP 3 | 151.65 / 152.41 | 18.53 / 18.59 | 88/88 | eos/eos |
| Corrected dense, no MTP | 155.21 / 155.30 | 14.80 / 14.92 | 96/96 | limit/limit |
| Indexed CSA, fixed MTP 3 | 152.80 / 152.58 | 16.87 / 17.14 | 91/91 | eos/eos |
| Indexed CSA, no MTP | 153.74 / 155.21 | 13.96 / 13.94 | 91/91 | eos/eos |

Every 31K run retrieved both facts. Corrected dense target-only reached its 96-token response limit; the other rows ended normally. The different output lengths and accepted draft counts matter when comparing TG.

| Mode, 59999 input tokens | PP, tokens/s | TG, tokens/s | Accepted / generated draft tokens |
| --- | --- | --- | --- |
| Legacy dense, fixed MTP 3 | 128.64 | 22.80 | 36 / 42 |
| Corrected dense, fixed MTP 3 | 128.38 | 21.54 | 35 / 45 |
| Indexed CSA, fixed MTP 3 | 132.76 | 24.15 | 37 / 39 |
| Indexed CSA, no MTP | 127.02 | 13.60 | none |

All 60K rows retrieved eight facts and produced 50 output tokens with normal end of response. Each row is one measured request.

| Mode, 142980 input tokens | PP, tokens/s | TG, tokens/s | Output tokens |
| --- | --- | --- | --- |
| Indexed CSA, no MTP | 133.68 | 12.35 | 50 |
| Indexed CSA, fixed MTP 3 | 131.53 | 21.34 | 50 |

Both 143K rows retrieved all eight facts, produced identical token IDs and ended normally. Dense modes were measured through 60K input with the same 144K allocation; no dense 143K speed is claimed.

The repeated 31K CSA prefill decreased from 487.05 / 487.55 seconds to 203.07 / 203.35 seconds, about 2.40x faster. Fixed-MTP TG changed from 12.66 / 12.46 to 16.87 / 17.14 tokens/s, with 93 versus 91 generated tokens.

Adaptive MTP on the same 31K prompt used a 256-token response budget and completed normally in 91 tokens: PP 152.75, TG 18.59 tokens/s, 49 / 73 accepted/generated draft tokens over 41 draft calls. Fixed MTP generated 108 draft tokens over 36 calls. The adaptive result differs from target-only by one synonymous word after 87 common token IDs; it is not a bit-exact match. This is one measurement, not a general adaptive-throughput claim.

Do not compare these 144K-allocation figures with older 64K-allocation measurements as though placement were identical. A larger KV/workspace reservation can move more experts to CPU even for a 31K request. TG also depends on the response length and accepted draft tokens; a short structured answer is not directly comparable to a longer explanation.

## Correctness after acceleration

The production build passed the independent oracle on CPU, Ada and Ampere, including shuffled cache slots, two sequences in a unified cache, long-position indexer RoPE and early queries at positions 0, 7 and 8 with a populated large cache. Selected indices matched exactly. Indexed main attention required NMSE below 1e-6 and the documented value-scaled F16 absolute bound. The early one-key case was checked separately.

Existing backend attention coverage passed 15 indexed cases and 12 dense 192-wide cases on each tested GPU architecture. Dense cases include token generation, multi-token verification and prefill with different GQA ratios. The maskless stream-stride fix also has a 512-dimensional multi-stream regression case.

Cache checks passed in separated and unified modes. Serialized state survived the round trip with all 23641724 bytes unchanged. F16 reduction order after cache compaction produced a maximum logit difference of about 0.0011284 and NMSE below 5.56e-6, with the same greedy token. The cache checker requires absolute error below 0.002 and NMSE below 1e-5; verification checks retain the tighter absolute 0.001 threshold and also require greedy agreement. All 12 rejection/acceptance cases passed in both cache modes. These bounds apply to the small deterministic model, not arbitrary full-model logits.

New fixed-MTP CSA and target-only token IDs matched on the tested 7K, 15K, two 31K repetitions, 60K and 143K requests. Adaptive MTP matched at 7K; its 31K result changes only "requested" to "required" near the end of the explanation. Both facts remain correct, all 91 tokens fit, and the response ends normally. Corrected dense also diverges between target-only and MTP after 51 common tokens; its target-only explanation reaches the 96-token output cap. These observations do not prove bit-exact full-model greedy behavior on arbitrary requests. CSA remains explicitly experimental and opt-in.

Tool and vision integration checks used a required-tool round trip, a synthetic red square, OCR of rendered digits 483726, and a subsequent text request returning 56 with MTP active again. Both new CSA target-only and MTP launches passed. The accelerated target-only 143K run also passed two cached repeats (142979 cached tokens, one new token) and a follow-up (143029 cached tokens, 33 new prompt tokens). Earlier adaptive-MTP cache checks passed at 60K/143K; the byte-level and numerical cache tests were repeated on the accelerated build.

## Incremental region summaries

CSA now reuses unchanged eight-token region summaries between microbatches. Only regions affected by KV writes, removals or other cache edits are recomputed. The BF16/E4M3 arithmetic and provider selection budget are unchanged. This removes repeated compression of the old prefix; query-dependent score GEMMs, top-k selection and attention still grow with their input sizes.

The summary cache adds about 205.49 MiB across the 23 full-attention layers at 144128 allocated tokens. Auto-fit accounts for it, so compare performance with a fixed weight placement. Set `LLAMA_STEP5_CSA_SUMMARY_CACHE=0` for the previous uncached summary path. Separate physical sequence streams and single-sequence unified caches use this optimization; a multi-sequence unified cache retains the reference path.

CPU checks compared cached and uncached operations on F32/F16/BF16 synthetic K/Z, including 26000-token histories, cache hits, one-region changes and incomplete regions. Small-model F16/Q8_0 checks produced identical logits for append, speculative verification/rejection, prefix replacement across the 4096-token fast-path boundary, sequence copies and full/sequence state restores. Full-checkpoint CUDA prefill speedups are not established by these CPU checks.

## Region scoring and top-k (2026-10-18)

The persistent summary cache now uses F16 instead of F32. E4M3 summary values are exact in F16, so this changes neither their values nor the score formula. The cached summary operation emits F16 directly and score GEMM reads that output without a whole-summary conversion kernel. For a single physical stream, that output aliases the persistent cache, so cache hits also avoid copying the summary. Multiple physical streams use a separate contiguous output. The F32 reference path remains available. Scratch allocation omits the converted-summary array for F16 input.

CUDA selection uses four-bit radix histograms rather than 32 one-bit scans. Warp aggregation reduces histogram atomics on supported NVIDIA devices. Selection exits early if the remaining bucket is entirely selected. The final gather retains ascending region IDs and the existing lower-ID tie rule; the budget is still 512 regions. Score GEMM and provider reduction remain separate.

F16/F32 score and selected-index checks matched exactly on CPU, logical CUDA4 (Ada) and logical CUDA5 (Ampere). Top-k matched an independent stable-sort oracle in 55 synthetic cases on each of these backends, including signed zeros, infinities, tied scores, adjacent FP32 values and region counts through 131072. Small-model F16/Q8_0 cache lifecycle checks passed in six CPU configurations; CPU MTP checks also covered host/backend media inputs, 4233-token histories and three-head drafting. These numerical checks do not establish a full-checkpoint prefill speedup. Dedicated checkpoint measurements are recorded below.

A paired CUDA-event microbenchmark on logical CUDA4 used identical synthetic scores for both kernels, 512 queries, 512 selected regions, six alternating rounds and 30 kernel calls per timing sample. Selected IDs matched exactly. The production server remained active, so these are shared-device kernel diagnostics, not dedicated full-model throughput measurements.

| Regions | Approximate history tokens | One-bit radix, ms | Four-bit radix, ms | Kernel speedup |
| --- | --- | --- | --- | --- |
| 3250 | 26000 | 0.17795 | 0.08912 | 2.00x |
| 7500 | 60000 | 0.45158 | 0.27930 | 1.62x |
| 18016 | 144128 | 1.32681 | 0.66092 | 2.01x |

The table uses random signed scores. All-equal scores measured 1.54x, 1.31x and 1.82x respectively. Standalone backend timings varied with concurrent inference; an apparent 60K regression in separate-process timings was absent in the alternating CUDA-event comparison. Compute Sanitizer reported zero memory errors and zero race hazards on the tested 513/3250-region boundary and tie cases.

## Full-checkpoint summary and top-k comparison (2026-10-18)

This comparison uses the IQ3_M-AntirezLike-AttnQ8_0-MTPQ6_K checkpoint, rather than the older TypeSafeAI Q3_K_M checkpoint used above. All six GPUs were dedicated to testing. Device order remained `2,1,0,5,4,3`. The CPU BF16 vision projector was loaded. Context was 144000 (144128 allocated), batch 2048, microbatch 512, 20 inference/batch threads, Q8_0 main K/V, fixed MTP 3 and `LLAMA_MTP_ADAPTIVE=0`. One placement was fitted with the larger F32 summary cache and a 512 MiB device margin, then reused with fitting disabled for all three profiles. Model-buffer sizes on every GPU and CPU matched exactly.

Each profile received the same warmup, then two complete uncached evaluations of the same 25998-token distributed retrieval prompt. Each answer contained 50 tokens and all eight requested facts. All six output token-ID sequences matched exactly, with 37 accepted draft tokens out of 42 generated in every request. No input was truncated. Each test server stopped normally. Model-loading time is excluded from the server's PP and TG figures.

| CSA profile | PP, tokens/s, two runs | TG, tokens/s, two runs |
| --- | --- | --- |
| Uncached summaries, one-bit radix | 251.99 / 247.84 | 30.16 / 29.92 |
| F32 summary cache, one-bit radix | 251.60 / 249.39 | 29.82 / 30.65 |
| Direct F16 summary cache, four-bit radix | 250.16 / 250.97 | 29.58 / 28.93 |

Mean PP changed from 249.91 to 250.57 tokens/s (+0.26%), within the observed run-to-run spread. This does not demonstrate a useful full-model prefill speedup at 26K. Mean TG for the short 50-token answer changed from 30.04 to 29.26 tokens/s (-2.60%). Two short responses per profile do not establish the cause or predict long-generation throughput; the lower observed TG must not be presented as a speedup.

Dedicated alternating CUDA-event top-k checks still measured about 1.87x / 1.61x / 1.98x for 512-query random-score batches at 3250 / 7500 / 18016 regions. At 3250 regions, one- and four-query batches measured 1.25x and 1.27x. At 18016 regions, some one-query or all-equal-score cases were up to about 6% slower. These are individual-kernel measurements, not full-model throughput claims. Selected IDs matched exactly.

The optimized runtime remains in an isolated local build. The normal `build-aliceai/bin` libraries were not replaced after this comparison because the full-model result does not justify a performance recommendation. The source changes retain the smaller summary cache and accelerated prefill selection for further profiling. No new full-model 60K or 144K result is claimed for this optimization.

## Memory and remaining costs

The allocation and profiling measurements below predate incremental summary caching.

The common saved placement has these loader-reported allocations for accelerated CSA with fixed MTP. Values are MiB; target KV includes main, SWA and indexer caches. MTP adds 193.24 MiB of model weights, 9 MiB of KV and 71.01 MiB of workspace on logical CUDA5.

| Logical GPU | Physical index | Model, including MTP | Target KV | Target workspace |
| --- | --- | --- | --- | --- |
| CUDA0 | 2 | 41919.97 | 2502.12 | 969.01 |
| CUDA1 | 1 | 43057.60 | 1880.34 | 971.27 |
| CUDA2 | 0 | 42183.78 | 2493.12 | 971.27 |
| CUDA3 | 5 | 43057.60 | 1880.34 | 968.97 |
| CUDA4 | 4 | 41851.74 | 2490.12 | 968.97 |
| CUDA5 | 3 | 17124.97 | 3123.91 | 1001.62 |

CPU model weights use 44365.30 MiB. The separate CPU projector is 4195.47 MiB, plus compute buffers and caches. Main full-attention KV is 9711.75 MiB; SWA KV is 207 MiB; indexer-cache allocation is 4451.22 MiB (3237.25 MiB of canonical BF16 K/Z and 1213.97 MiB of currently unused V storage). Corrected/legacy dense do not allocate the indexer cache and reserve about 312-334 MiB of target workspace per GPU, versus about 969-1002 MiB for CSA. Fixing weight placement leaves that memory available in dense instead of moving additional experts to GPU.

| Profile | Observed startup to listening, s | Sampled process high-water RSS, GiB |
| --- | --- | --- |
| Direct CSA, fixed MTP | 325.92 | 50.37 |
| Legacy dense, fixed MTP | 322.81 | 52.25 |
| Corrected dense, fixed MTP | 323.71 | 52.25 |
| Corrected dense, target only | 324.23 | 51.04 |
| Indexed CSA, fixed MTP | 332.17 | 56.31 |
| Indexed CSA, target only | 331.27 | 56.52 |
| Indexed CSA, adaptive MTP | 323.72 | 50.48 |

Load figures include the CPU projector and differ in disk/page-cache state; these are observed startup durations, not controlled cold-load comparisons. Whole-device peak used VRAM in the accelerated fixed-MTP profile was approximately 44.9-45.5 GiB on the Ada devices and 21.4 GiB on the RTX 3090. These whole-device samples include CUDA overhead and are not the same as the model-buffer sizes above.

A separate warmup capture used one 2048-token decode batch at history 4096 with microbatch 512, without MTP. The probe synchronized before/after the capture. It recorded CPU subgraphs and CUDA activity on all six GPUs. GPU collection and report export ended before the timed repetitions; CPU timing instrumentation remained enabled throughout the sweep.

| Captured component | Duration, ms | Interpretation |
| --- | --- | --- |
| Complete synchronized decode | 13649.98 | Wall elapsed time |
| CPU subgraphs containing MoE | 6605.87 | Elapsed time summed over complete CPU subgraphs; 176 MUL_MAT_ID nodes |
| Other CPU subgraphs | 48.46 | Elapsed time |
| GPU matrix/dequantization kernels | 3082.43 | Includes MoE, dense projections and indexer GEMMs |
| GPU indexed CSA attention | 288.75 | 92 calls: 23 layers x 4 microbatches |
| GPU dense/SWA attention | 67.02 | 276 calls: 69 layers x 4 microbatches |
| GPU indexer kernels | 24.66 | Norm/compression/postprocessing/top-k/row mapping; GEMMs counted above |
| GPU copy/gather kernels | 16.92 | Kernel execution time |
| Other GPU kernels | 183.31 | Kernel execution time |
| Host-to-device copies | 2453.30 | 437 copies, 5041.706 MB |
| Device-to-host copies | 449.22 | 301 copies, 1734.095 MB |
| cudaLaunchKernel | 50.64 | Host API elapsed time over 22916 calls |
| CUDA graph launch | 0.94 | Host API elapsed time over 57 calls |
| cudaStreamSynchronize | 6794.52 | Host wait time; overlaps device activity |

CPU process time during MoE-containing subgraphs was 113701.41 ms across all process threads. It is not exclusive time in one MoE function. GPU durations sum events across devices; CPU subgraphs, copies, kernels and synchronization can overlap. These rows must not be added as disjoint percentages of decode wall time. The capture is one synthetic short-history warmup point, not a profile of the complete 143K request.

After attention acceleration, CPU MoE, GPU matrix kernels and host/device transfers remain substantial costs. Host launch-call time is much smaller at this point. Changing placement or microbatch can affect all those terms, so the working configuration is validated separately.

CSA retains original main K/V and canonical BF16 indexer K/Z. Unchanged regions now reuse persistent F16 summaries; edits invalidate their versions. Long-context KV memory still grows with context size. Auto-fit reserves the target cache, indexer cache, derived summaries, MTP cache and graph workspace together.

## Microbatch sweep

| Microbatch | Mean PP, tokens/s | Sample standard deviation | Three measured samples, tokens/s |
| --- | --- | --- | --- |
| 256 | 123.21 | 0.25 | 123.37 / 122.92 / 123.34 |
| 512 | 145.67 | 6.70 | 150.81 / 138.09 / 148.11 |
| 1024 | 176.72 | 0.17 | 176.91 / 176.64 / 176.60 |
| 2048 | 202.28 | 1.26 | 202.57 / 203.36 / 200.90 |

The actual server validation regenerated auto-fit for each candidate at context 144000 and the same 2048 MiB margin. Placement changes are part of the configuration comparison; these are not pure microbatch-only kernel measurements. Each candidate used a 7K warmup, two uncached 31K requests and one uncached 60K request, fixed MTP 3 and the same saved prompts.

| Microbatch | CPU model weights, MiB | 31K PP, tokens/s | 31K TG, tokens/s | 31K output / stop | 60K PP / TG, tokens/s |
| --- | --- | --- | --- | --- | --- |
| 512 | 44365.30 | 152.80 / 152.58 | 16.87 / 17.14 | 91 / eos | 132.76 / 24.15 |
| 1024 | 50371.30 | 161.70 / 161.25 | 18.22 / 18.25 | 96 / limit | 144.72 / 19.28 |
| 2048 | 62383.30 | 154.54 / 154.40 | 14.02 / 14.09 | 91 / eos | 143.29 / 16.67 |

All candidates retrieved both 31K facts and all eight 60K facts without input truncation. Microbatch 1024 reached the 96-token explanation limit in both 31K runs; its TG is not an equal-output comparison. Microbatch 2048 returned the same 91-token answer and the same draft counts as 512 at 31K, but its TG fell from about 17.0 to 14.1 tokens/s. The 60K responses had 50 tokens and normal completion in every candidate, with different draft acceptance.

Both 1024 and 2048 passed the accelerated fixed-MTP 60K cached repeats/follow-up, required-tool round trip, CPU red-square/OCR checks and text-after-vision request. Microbatch 512 remains the working baseline: it has the broadest paired coverage through 143K and preserves TG. Microbatch 1024 gives a modest prefill alternative with more CPU offload and lower 60K TG; 2048 is not recommended for this 144K allocation. Auto-fit must be rerun if microbatch, context or visible-device order changes.

The standalone `llama-bench` sweep uses 8192 synthetic random token IDs, batch 2048, the same saved main-model placement, no MTP, and three repetitions for each microbatch 256, 512, 1024 and 2048 after a full warmup. Each invocation draws a deterministic random-token subsequence from C std::rand; prompts differ across cases and repetitions; it is a synthetic tuning check, not a same-prompt paired quality test. Its smaller context allocation also differs from the server's 144K allocation. A larger microbatch needs validation with actual 144K auto-fit before changing the working launcher.

## Reproduction

Build with `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, Release and architectures appropriate to the devices. Use [the launch script](../../examples/step5/run-server.sh) and preserve the actual auto-fit arguments when comparing modes. For example:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,1,0,5,4,3
export STEP5_MODEL=/path/to/Step5-Preview-Q3_K_M.gguf-00001-of-00008.gguf
export STEP5_MMPROJ=/path/to/mmproj-Step5-Preview-bf16.gguf
export STEP5_ATTENTION=csa
export STEP5_CONTEXT=144000
export STEP5_UBATCH=512
export STEP5_MTP=3
export LLAMA_MTP_ADAPTIVE=0
./examples/step5/run-server.sh
```

After readiness, `python3 examples/step5/check-server.py --url http://127.0.0.1:8085 --tokens 60000 143000 --follow-up --output step5-quality` generates mixed archives, records tokenizer-measured lengths and validates distributed facts, prefix reuse and a follow-up. Set adequate output budgets; prompt plus output must fit. Restart with `STEP5_MTP=0` for target-only comparison, `STEP5_ATTENTION=dense` for corrected dense, or `STEP5_ATTENTION=legacy` for the earlier scale/norm convention. Pass `--spec-type` only once.

For repeatable speed comparisons, keep requests and their tokenization fixed, disable prompt-cache reuse for each measured prefill, save all output token IDs and report both repetition count and output length. Resource samples report whole-device used VRAM and process host RSS, which are different quantities from the loader's model/cache/workspace buffers. Load times depend on disk and page-cache state and must not be treated as controlled cold-disk benchmarks.


## Separate MTP Q8_0 on the new Q3 target (2026-10-17)

The retained official FP8-to-BF16 source was used to quantize all 30 matrices of the three MTP heads to Q8_0. A control sidecar copied the existing Q6_K heads without requantization. Both sidecars preserved the Q3 target embeddings, norms and RoPE factors exactly. Full files contain 54 tensors: Q6_K is 2,502,333,184 bytes and Q8_0 is 3,111,627,136 bytes.

The target was the new `Q3_K_M-AttnQ5_K-MTPQ6_K`, with CSA, context 144000, microbatch 512, Q8_0 target/draft K/V, three heads and adaptive drafting disabled. The same target placement, fitted once for Q8_0, was used for both formats. Target MoE weights were partly on CPU; the sidecar heads ran on logical CUDA5 (RTX 3090), with `CUDA_VISIBLE_DEVICES=2,1,0,5,4,3`. The background quantization service was frozen during measurement and resumed afterward. No projector was loaded.

| Case | Q6_K tok/s | Q8_0 tok/s | Change | Q6 acceptance | Q8 acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 25.60 | 25.25 | -1.4% | 85.6% | 86.8% |
| explanation | 19.80 | 18.79 | -5.1% | 54.9% | 55.4% |
| retrieval-31k | 19.44 | 18.05 | -7.2% | 62.4% | 59.4% |

Code and explanation had two repeats per format; retrieval had one. All five paired generated token sequences were identical, and both retrievals returned the required facts. Aggregate TG was 21.79 versus 20.91 tok/s (-4.1% for Q8_0), with draft acceptance 66.20% versus 66.56%. Q8_0 did not improve speed in this finite test. These measurements compare two external sidecars against the same Q3 target; they do not establish results for Antirez-like Q2 or the embedded-MTP configuration.

All three heads passed CUDA decode and byte-exact sequence-state roundtrip checks for both formats. A full target without CSA tensors was still rejected. Both test servers were stopped after measurement.

## CSA Q8_0 asynchronous loading (2026-10-17)

The 192-wide indexed Q8_0 CUDA kernel now preloads compressed rows into a single shared-memory buffer. Loading V overlaps Q x K; loading the next K overlaps attention x V. Cache tensor sizes and BF16 indexer storage are unchanged. The extra buffer occupies 6.5 or 13 KiB of shared memory per CUDA block.

These are operator microbenchmarks without a model checkpoint. The same test binary loads either the previous synchronous CUDA library or the new asynchronous library. Each case uses 32768 physical cache rows, 4352 selected rows, head width 192 and 64 query heads across four provider streams. Selection patterns and shapes match; tensor values are independently initialized. CUDA graphs are enabled. Times are medians of four alternating paired runs, each measuring at least one second; they include the native backend's graph submission and synchronization. The background CPU quantization service remained active throughout.

| GPU | Queries | Before, us | After, us | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Ada RTX 4090 | 1 | 43.90 | 36.10 | 1.22x |
| Ada RTX 4090 | 3 | 71.22 | 54.54 | 1.31x |
| Ada RTX 4090 | 512 | 8522.78 | 6457.06 | 1.32x |
| Ampere RTX 3090 | 1 | 90.75 | 58.67 | 1.55x |
| Ampere RTX 3090 | 3 | 224.17 | 130.39 | 1.72x |
| Ampere RTX 3090 | 512 | 24340.31 | 18006.61 | 1.35x |

F16 controls differed by at most 2.1% between library profiles. Q8_0 remains slower than F16 in these fixtures. These results measure Q8_0 KV-cache loading and attention, independently of MTP weight precision. They do not establish an end-to-end TG or prefill improvement for the full model.

Both architectures passed 20 independent Q8_0 oracle cases, 20 F16 controls and 23 native indexed attention cases. Selected indices matched exactly; maximum Q8_0 attention NMSE was 7.41e-7, within the existing tensor-core tolerance. The small CSA model also passed cache restore, prefix copy, independent sequences and speculative rollback in both separate and unified cache modes.

CUDA Compute Sanitizer memcheck and racecheck passed all seven 192-wide Q8_0 indexed cases on each architecture, including partial tiles and two-byte-offset views. Both racecheck runs reported zero errors and zero warnings.

To reproduce the operator timing after building `test-backend-ops`:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,1,0,5,4,3
./build-aliceai/bin/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT -p 'head_dim=192' --output sql
```

Use logical CUDA5 for the Ampere measurement. Compare old/new libraries with the same binary and retain the actual `time_us` values; this test tool's CSV format omits timing fields.

## MTP with images retained in history

The IQ3_M-AntirezLike-AttnQ8_0-MTPQ6_K runtime passed eight CUDA checks using the BF16 projector on CPU, CSA, Q8_0 main K/V, F16 draft K/V, context 144000, batch 2048, microbatch 512 and the default adaptive three-head MTP budget. CUDA device order was `2,1,0,5,4,3`. Each history recognized a red square, retained it for a text follow-up, added a blue circle, then identified both colors. All requests kept MTP enabled and completed without cache consistency resets.

The text follow-up listed integers 1 through 30 and generated 89 tokens with greedy sampling. The image remained in the prompt in both measurements:

| Text prefix before image | Reused prompt tokens | New prompt tokens | TG (tokens/s) | Accepted / drafted |
| --- | --- | --- | --- | --- |
| 4594 | 4789 | 39 | 31.57 | 67 / 69 (97.1%) |
| 25996 | 26191 | 39 | 30.67 | 65 / 68 (95.6%) |

These measurements apply to the saved counting requests. Full-model target-only speed and token-ID equivalence were not compared in this run. Synthetic CPU checks separately matched all 60 generated token IDs against target-only decoding, including retained-image cache reuse and a second image. The temporary CUDA server was stopped after validation.
