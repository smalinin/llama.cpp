# Step-5 CUDA performance validation

These are local measurements on the restored TypeSafeAI Q3_K_M checkpoint with 184 NeuroSenko BF16/F32 indexer tensors. They do not measure a newly converted official checkpoint or establish a broad quality score. The implementation and numerical contract are described in [step5.md](step5.md).

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

## Memory and remaining costs

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

CSA retains original main K/V. It stores canonical BF16 indexer K/Z and recomputes region summaries in the graph rather than maintaining a separate persistent summary state. Consequently, attention arithmetic falls substantially while long-context KV memory still grows with context size. Auto-fit must reserve the target cache, indexer cache, MTP cache and graph workspace together.

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
