# Step-5 Preview

Step-5 uses the STEP35 GGUF architecture: 92 decoder layers and three appended MTP layers, hidden width 4096, 64 query heads, four KV heads, head width 192, and a repeating pattern of three 512-token sliding layers followed by one sparse layer. There are 352 routed experts and eight active experts per token.

The reference is [StepFun's standalone implementation](https://huggingface.co/stepfun-ai/Step-5-Preview/tree/364b3b03fed2477aa0953fe855a2c4c8c48ee3cc/inference), revision `364b3b03fed2477aa0953fe855a2c4c8c48ee3cc`. The original preview GGUF can be used with restored indexer tensors; matching dimensions does not establish identical weights to the official release.

## CUDA build and launch

Complete inference was developed and validated in a CUDA configuration with CPU expert offload and an optional CPU projector. Full CPU-only execution, other GPU backends and GPU vision have not been validated. Numerical CPU operator coverage is separate from complete-model validation.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j 8 --target llama-server llama-cli llama-bench

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Set the visible devices to the order appropriate to your machine.
# The validation used CUDA_VISIBLE_DEVICES=2,1,0,5,4,3.
export STEP5_MODEL=/path/to/Step5-Preview-Q3_K_M.gguf-00001-of-00008.gguf
export STEP5_ATTENTION=csa
./examples/step5/run-server.sh
```

Keep all eight restored parts in the same directory. The original seven-part TypeSafeAI release lacks required indexer tensors. The validated Ampere/Ada build additionally used `-DCMAKE_CUDA_ARCHITECTURES="86-real;89-real"`; use architectures that match the actual devices. `GGML_NATIVE=ON` targets the build machine's CPU.

The script uses context 144000, batch 2048, microbatch 512, F16 K/V, Flash Attention, eager CPU weight loading (`--load-mode none`), 20 CPU threads, one slot and disabled context shifting. It selects auto-fit with a 2048 MiB device margin and the requested minimum context. Context, batch, microbatch, threads and model/binary paths are configurable through `STEP5_CONTEXT`, `STEP5_BATCH`, `STEP5_UBATCH`, `STEP5_THREADS`, `STEP5_THREADS_BATCH`, `STEP5_MODEL` and `STEP5_SERVER`.

Set `STEP5_CACHE_TYPE=q8_0` to use Q8_0 for the main model K and V, including CSA. Draft K/V have separate options described below. The default remains `f16`. Direct CLI launches can use `--cache-type-k q8_0 --cache-type-v q8_0 -fa on`. CSA requires matching F16/F16 or Q8_0/Q8_0 caches; mixed formats and other cache types are rejected. Indexer K/Z remain canonical BF16. The CUDA indexed path dequantizes selected Q8_0 blocks into shared-memory tiles for the existing tensor-core attention kernel, without a full F16 copy of the KV cache. CPU fallback supports Q8_0; explicit FP32 indexed attention uses a dequantized K row and the original FP32 query. On Ampere and newer CUDA devices, the 192-wide Q8_0 indexed kernel preloads compressed tiles with `cp.async`: V loading overlaps Q x K, and the next K loading overlaps attention x V. A single compressed shared-memory buffer adds 6.5 or 13 KiB per block; there is no additional global cache copy. Devices without asynchronous-copy support retain synchronous reads. On Ada and Ampere, the independent numerical oracle, native indexed attention and small-model KV lifecycle checks passed. See [operator timing results](step5-performance.md#csa-q8_0-asynchronous-loading-2026-10-17) for the measured speedup; full-model throughput has not been compared for this loading change.

For one slot, context 131072, microbatch 512 and three MTP heads, cache tensor storage is about 12.79 GiB with F16 or 8.14 GiB with Q8_0 in CSA mode. This includes the indexer cache and its unused V allocation, but excludes weights, projector, graph workspace and backend overhead. Quantized K/V can change logits and generated text. Re-run auto-fit after changing the cache type; the numerical oracle is not a full-model quality benchmark.

CSA remains opt-in. The launcher defaults to corrected dense unless `STEP5_ATTENTION=csa` or the existing CSA environment flag is set. `STEP5_ATTENTION=legacy` selects the earlier dense scale/norm convention. The obsolete token-sparse experiment is not a launcher mode. Each mode sets its own runtime flags; explicit mode selection clears conflicting inherited flags.

MTP defaults to three heads with existing adaptive draft length. Use `STEP5_MTP=0` for target-only execution, or `LLAMA_MTP_ADAPTIVE=0` for a fixed draft budget. The launcher passes `--spec-type` once, including when raw arguments override it. Functional MTP checks passed; adaptive decoding can change a synonymous word relative to target-only in the tested 31K explanation. The fixed-budget paired fixtures at microbatch 512 matched through 143K. These finite tests do not establish arbitrary-request bit-exact greedy behavior.

A separate Step-5 MTP GGUF can be selected with `--model-draft /path/to/Step-5-Preview-MTP-Q8_0.gguf --gpu-layers-draft all` appended to the launcher command. The loader accepts MTP-only files that retain all three heads, token embeddings, output norm and optional RoPE factors. Main-model MTP weights are skipped when an external draft is selected. The target still requires its CSA tensors. An MTP-only file is used as a draft, not as the target model.

The Q8_0 sidecar changes the 30 MTP matrices to Q8_0; its embeddings and norms are copied from the Q3 target for a controlled comparison with the extracted Q6_K sidecar. To select Q8_0 K/V for the draft as well, pass `--spec-draft-type-k q8_0 --spec-draft-type-v q8_0`. Weight quantization and KV cache type are independent.

For image requests, set `STEP5_MMPROJ=/path/to/mmproj-Step5-Preview-bf16.gguf`. The launcher adds `--no-mmproj-offload`, keeping the projector in RAM on CPU. Without a projector, use text/tool requests. Step-5 MTP accepts projected image embeddings and stays enabled when images remain in the conversation history. The projector still runs on CPU; MTP can accelerate text generation but does not accelerate image encoding.

The default address is `http://127.0.0.1:8085`, with the OpenAI-compatible API under `/v1`. `STEP5_HOST` and `STEP5_PORT` override it. Stop an existing server before starting another full copy. Prompt plus response must fit within the selected context. The largest checked initial prompt is 142980 tokens with microbatch 512; advertised 1M context and official-release weight identity remain unverified.

## Tensor compatibility

Each of the 23 full-attention layers requires these tensors. Dimensions below use GGML order.

| Source suffix under self_attn | GGUF suffix under blk.N | Dimensions | Convention |
| --- | --- | --- | --- |
| sparse_indexer_q.weight | indexer.q_proj.weight | 4096, 4096 | Raw BF16 |
| sparse_indexer_k.weight | indexer.k_proj.weight | 4096, 256 | Raw BF16 |
| sparse_indexer_z.weight | indexer.z_proj.weight | 4096, 256 | Raw BF16 |
| sparse_indexer_w.weight | indexer.proj.weight | 4096, 16 | Raw FP32 |
| sparse_indexer_q_norm.weight | indexer.q_norm.weight | 256 | Zero-centered RMSNorm: multiply by 1 + weight |
| sparse_indexer_k_norm.weight | indexer.k_norm.weight | 256 | LayerNorm scale, no offset |
| sparse_indexer_k_norm.bias | indexer.k_norm.bias | 256 | LayerNorm bias |
| ssmax_s | attn_ssmax_s | 64 | Preserve for checkpoint compatibility; unused by the reference |

The TypeSafeAI Q3_K_M model with the NeuroSenko restoration has eight splits, 1783 tensors, and all 184 required indexer tensors with the expected dimensions. Its main RMSNorm tensors already include the conversion offset. Restored indexer norms retain source values. Do not add the offset to both the stored tensor and the runtime expression.

The converter recognizes `Step5ForCausalLM`, `Step5ForConditionalGeneration`, and the earlier preview architecture names. Official model configs nest the text configuration under `text_config`.

## Reference attention

CSA compresses completed eight-token regions using a softmax over Z independently for each proxy coordinate. K is normalized and position-rotated first; Z is not normalized or rotated. Four provider groups each use four indexer query heads and select up to 512 historical regions, plus the query's current region. Main attention reads original token-level K/V with a causal mask.

Indexer RMSNorm is zero-centered; indexer LayerNorm uses the configured RMS epsilon. The reference keeps `ssmax_s` as an unused parameter and uses the usual inverse-square-root head scale for attention. Region-summary and score-input E4M3 rounding, and ties resolved by smaller logical region ID, are part of the reference contract.

The implementation below follows this contract. Token-level top-4096 selection is a separate legacy experiment and is not equivalent to regional CSA.

## Fused CSA score calculation

The combined CUDA score kernel is enabled by default for CSA prefill batches with at least 32 queries. Set `LLAMA_STEP5_CSA_FUSED_SCORES=0` before launch to use the previous cuBLAS path. Smaller decode/MTP batches keep the cuBLAS path in either mode. This changes neither the 512-region budget nor the attention mode.

On Ampere/Ada, the kernel uses the existing FP16 tensor-core primitives with FP32 accumulation. Each block processes 64 regions and eight queries. Per-head ReLU and the ordered weighted sum of four heads are applied before writing the four provider scores; the 16-head dot-product matrix is not written to global memory. Query E4M3 conversion stays separate. Both F16 and F32 summaries are accepted, and the CPU implementation keeps its reference score arithmetic.

With 144128 allocated tokens and microbatch 512, the score backing buffer, including scratch and allocation padding, decreases from 844.50 to 144.87 MiB. This is an operator-buffer calculation, not the reduction in total process VRAM. Auto-fit accounts for the smaller graph allocation; rerun fitting when changing the flag. Compare fixed placement first to separate the kernel change from moving more experts to GPU. On the tested IQ3_M checkpoint at 26K, fixed-placement prefill was unchanged; rerunning auto-fit moved 2.66 GiB more weights to GPU and improved mean prefill by about 9%. See [the performance report](step5-performance.md) for CUDA numerical checks and measurements.

## Numerical oracle

Use `--kv-type q8_0` with `examples/step5/check.py` to compare Q8_0 cache reads and attention against independently dequantized GGUF blocks. The default `--kv-type f16` retains the F16 fixtures. Both formats require exact agreement on selected token indices. Add `--fused-scores` to validate the compact score operation; this checker option also exercises short query batches that the runtime leaves on cuBLAS.

Run `python examples/step5/reference.py --output report.json --fixtures fixtures` in a Python environment with PyTorch and NumPy. This uses CPU calculations and does not load the model. It checks dense equivalence before the region budget is exceeded, causal selection, provider isolation, zero-centered norms and stable ties. Binary fixtures provide independent expected results for backend validation.

## Attention compatibility mode

Step-5 uses the reference attention scale by default. Set `LLAMA_STEP5_LEGACY_ATTENTION=1` to reproduce the previous `ssmax_s` scale and indexer normalization for a controlled dense comparison. This flag does not affect ordinary Step3.5 checkpoints. Indexer affine parameters, projections, normalization outputs and RoPE outputs are rounded to BF16 in the corrected path.

CPU normalization matched the independent PyTorch fixture exactly. CUDA on Ada and Ampere matched Q exactly; one of 32768 K values differed by one BF16 ULP (0.00390625), within the FP32 reduction tolerance. Existing GGML NORM, RMS_NORM and RoPE backend checks also passed on CUDA.

## Experimental grouped CSA

Set `LLAMA_STEP5_EXPERIMENTAL_CSA=1` to enable region compression and four independent provider groups. The older `LLAMA_STEP5_EXPERIMENTAL_SPARSE=1` still selects individual tokens and is a separate comparison mode. CSA requires matching F16/F16 or Q8_0/Q8_0 K/V and Flash Attention. Keep context shifting disabled.

The indexer uses BF16 projection/norm/RoPE boundaries and E4M3 score inputs. Precise CUDA trigonometry preserves the BF16 reference table at long positions despite the build's global fast-math flags. The attention path maps selected logical positions to physical rows and reads original K/V through native indexed Flash Attention, including multi-token verification batches. Unusual layouts retain a direct attention kernel fallback. Independent logical sequences in a unified physical cache use the indexed path.

Run `python examples/step5/check.py --libdir build/bin --backend CUDA0 --output csa.json` with PyTorch and NumPy; `CPU` and other logical CUDA device numbers are also accepted. Checks cover the reference boundary lengths, signed head weights, stable ties, shuffled cache slots, independent sequences, and 192-wide attention with 64 query heads. Norm/RoPE fixtures at positions 0, 4096 and 143999 matched the official Triton kernel exactly on Ada and Ampere. Selected indices matched the independent oracle exactly in all fixtures. Rare FP32 summary reductions can straddle a BF16 midpoint before E4M3; the validator reports those elements and requires them to stay in the quantization interval of a 1e-6 FP32 error bound.

The accelerated path also covers queries at positions 0, 7 and 8 with a populated 8192-slot cache. Native CUDA Flash Attention uses F16 probability and accumulator tiles; it is not an FP32-accumulation promise. Indexed attention requires NMSE below 1e-6 and a value-scaled absolute bound; the one-key case is checked separately. CPU indexed attention honors the explicit FP32 accumulator request. Numerical operator checks and full-model checks are separate.

CUDA top-k uses four-bit radix histograms, at most eight rounds, with early exit when the remaining bucket is entirely selected. It retains the same region budget, ascending region IDs and lower-ID preference for equal scores. The fused path described above combines score GEMM and provider reduction for prefill by default. Setting `LLAMA_STEP5_CSA_FUSED_SCORES=0` or using fewer than 32 queries retains separate operations.

## Cache and memory

CSA stores canonical BF16 K/Z token projections in the indexer cache. Region summaries are cached per layer and physical sequence stream. Each KV write or removal changes the affected eight-token region's version; unchanged completed regions reuse the same BF16/E4M3 result. Rejected MTP tokens invalidate their regions, including edits made while the dense short-context fallback is active. Prefix copies, cache clearing and state restore also invalidate the affected derived data. Summaries are rebuilt from canonical K/Z after restore and do not change the serialized state format. Future completed regions are excluded per query before selection.

Summary caching is enabled by default with CSA for separate sequence streams or a single unified sequence (including `--parallel 1`). Multiple independent sequences sharing one unified stream keep the uncached summary path. Set `LLAMA_STEP5_CSA_SUMMARY_CACHE=0` before launch to use the uncached reference for comparison. The cache adds 520 bytes per region per CSA layer: about 205.49 MiB total across 23 layers at 144128 allocated tokens. Auto-fit includes these persistent buffers. Query-dependent scores, top-k selection and attention still run on each microbatch. The persistent summaries use F16, which exactly represents their E4M3 values. CUDA GEMM reads the F16 summaries directly, avoiding a repeated F32-to-F16 conversion. For one physical stream, the result views the persistent cache and unchanged regions require no summary copy; multiple physical streams retain a contiguous gathered result. The uncached F32 summary API remains available for reference comparisons.

The logical slot map handles independent sequences and physical cache fragmentation. Chunked prefill may append to an existing cache. Context shifting remains unsupported. CSA scratch is part of the GGML graph allocation, including the cuBLAS score workspace, so auto-fit measures it. The actual logical range bounds normal execution; full-cache graph reservation still measures the context limit.

Generate the small test GGUF with `PYTHONPATH=gguf-py python examples/step5/make-small.py`. Build `examples/step5/check-cache.cpp` against libllama and run it with the GGUF path and either `0` (separate streams) or `1` (unified cache). It checks chunk sizes 128/256, prefix copy, independent sequences, verification rollback, state restore into a context with microbatch 512, and CUDA graph reuse. Both modes kept the same greedy token. All 23641724 serialized bytes matched exactly after restore. F16 reduction order after cache compaction produced a maximum logit difference of about 0.0011284 and NMSE below 5.56e-6. The checker requires absolute error below 0.002, NMSE below 1e-5 and the same greedy token. These are bounds for the deterministic four-layer fixture, not the full checkpoint.

The six-GPU auto-fit plan for context 144000 and microbatch 512 with a 2048 MiB margin was used for full-model checks through 142980 input tokens. Indexed attention generated exactly the same placement as the previous direct kernel. Saved main-model placement was fixed for target/MTP and attention-mode comparisons; memory planning and model quality were checked separately.

## MTP verification

The existing three-head MTP implementation is reused, including its adaptive draft length. No new draft model is needed. Use `--spec-type draft-mtp --spec-draft-n-max 3`; set `LLAMA_MTP_ADAPTIVE=0` for fixed-budget comparisons. Disable drafting with a separate `--spec-type none --spec-draft-n-max 0` launch. Pass `--spec-type` once: repeated options append types.

`check-cache.cpp MODEL 0 verify` and `check-cache.cpp MODEL 1 verify` check all 12 acceptance counts for verification batches of two, three and four tokens. Each crosses an eight-token region boundary. After rejection, a clean target context that never decoded the rejected tokens is the reference. Both cache modes passed, with equal greedy tokens and maximum absolute logit error below the explicit 0.001 verification threshold on the deterministic four-layer test model. The 0.001 tolerance covers batch-dependent arithmetic on this test model; it is not a bound for the full checkpoint.

The oracle and CPU/CUDA summaries also cover extreme Z values. Per-coordinate denominators use the reference clamp at 1e-20; an underflowing coordinate returns zero instead of NaN.

With the restored Q3_K_M weights, microbatch 512 and identical main-model placement, the accelerated CSA path produced identical target-only and fixed-MTP output token IDs for the tested 7382-, 15503-, 31028-, 59999- and 142980-token prompts. The 31K case was repeated twice in each mode and returned the same 91 tokens. This finite set does not establish bit-exact greedy behavior on arbitrary requests. Before indexed acceleration, the older direct kernel returned both 31K facts correctly but diverged in its explanation after 46 common tokens; that particular discrepancy is absent in the new fixed-MTP checks.

The accelerated adaptive-MTP run matched target-only at 7K. On the same 31028-token prompt, it completed in 91 tokens with both facts correct, but changed one word in the closing explanation ("requested" versus "required") after 87 common token IDs. Its 256-token output budget did not cut the response. It generated 73 draft tokens over 41 calls, versus 108 over 36 fixed-budget calls, confirming actual budget adaptation. Corrected dense also shows target-only/MTP differences on 31K, after 51 common tokens. Full-model bit-exact greedy behavior is not established; CSA remains opt-in while this numerical limitation is reviewed. Functional cache and fact checks passed. Changing the prefill microbatch also changes full-model greedy wording in the measured 1024 case; exact operator fixtures and byte-preserving state restore do not guarantee batch-independent full-model token IDs.

## Full-model quality checks

Validation used the existing restored TypeSafeAI Q3_K_M checkpoint, not a newly converted official checkpoint. Its eight GGUF splits contain 1783 tensors, including all 184 restored indexer tensors. Matching dimensions does not prove identity to StepFun's released weights or general long-context accuracy.

Complete inference was tested on CUDA 13 with five 48 GiB Ada GPUs and one 24 GiB RTX 3090, plus CPU expert offload. Every GPU test used `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=2,1,0,5,4,3`. Numerical CPU operator checks also passed. CPU-only complete-model inference and other GPU backends were not tested.

Long requests used context 144000 (allocated as 144128), batch 2048, microbatch 512, 20 CPU threads, F16 K/V, Flash Attention and disabled context shifting. Auto-fit generated placement with a 2048 MiB device margin; paired runs then fixed that placement. MTP uses the existing three draft heads; fixed and adaptive budgets are measured separately.

| Input tokens | Check | Result |
| --- | --- | --- |
| 7382 | Two archive facts | 2/2, identical target-only/fixed-MTP token IDs |
| 15503 | Eight archive facts | 8/8, identical target-only/fixed-MTP token IDs |
| 31028 | Two facts and requested explanation, two repetitions per mode | 2/2, all 91 token IDs equal with target-only/fixed MTP |
| 59999 | Mixed archive, eight distributed facts | 8/8, all 50 token IDs equal, normal end of response, no input truncation |
| 142980 | Mixed archive, eight distributed facts | 8/8, all 50 token IDs equal, normal end of response, no input truncation |
| 142980, cached, accelerated target | Repeated archive | 8/8; 142979 cached input tokens, one token processed again |
| Approximately 143K, accelerated target | Follow-up asking for THETA and ALPHA in reverse order | 2/2; 143029 cached tokens, 33 new prompt tokens |
| 59999, cached, accelerated fixed MTP, microbatch 1024/2048 | Repeated archive and follow-up | 8/8 and 2/2 in both configurations |
| 59999, cached, earlier adaptive MTP | Repeated archive and follow-up | 8/8 and 2/2; 59995 cached input tokens on repeat |
| 142980, cached, earlier adaptive MTP | Repeated archive and follow-up | 8/8 and 2/2; 142976 cached input tokens on repeat; follow-up retained 143076 tokens |

Mixed archives contain Russian prose, Python, JSON, tables, request logs and English documents. `examples/step5/check-server.py` generates them using the running server's tokenizer, saves complete requests and responses, rejects contradictory values, and optionally checks prefix reuse and a follow-up turn. For example, `python3 examples/step5/check-server.py --url http://127.0.0.1:8086 --tokens 60000 143000 --follow-up --output step5-quality`. Output budgets must fit within the selected context. Checks deliberately disable sampling for repeatability.

New accelerated target-only and fixed-MTP servers passed real required-tool round trips, including structured arguments, a tool-result message and the final answer 17. Both used string `tool_choice: "required"`. Named-object enforcement is not supported by this server path. Parallel tool calls and strict-schema reliability on arbitrary requests were not tested.

Image checks used the separate projector on CPU: synthetic red-square recognition and OCR of rendered digits 483726. A subsequent text-only request returned 56; MTP drafting resumed in the MTP launch. These fixtures do not cover natural-image accuracy, a GPU projector or video.

The IQ3_M-AntirezLike-AttnQ8_0-MTPQ6_K build also passed eight CUDA image-history requests with CSA, Q8_0 main K/V, a CPU BF16 projector, context 144000, microbatch 512 and the default adaptive three-head MTP budget. Text prefixes contained 4594 or 25996 tokens before the first image. Each history tested image recognition, a text follow-up retaining that image, a second image retaining the first, and a question about both. MTP stayed enabled in all requests, and follow-ups reused the cached prefix without target/draft consistency resets. These finite checks use two synthetic images; see [the performance report](step5-performance.md) for the saved counting-request measurements.

The maximum checked initial prompt was 142980 tokens; the earlier adaptive-MTP follow-up retained 143076 tokens. No 1M-context validation is claimed. Performance comparisons control warmup, placement and prompt-cache reuse separately; see [the CUDA performance report](step5-performance.md).
