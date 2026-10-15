# Step-5 Preview

Step-5 uses the STEP35 GGUF architecture: 92 decoder layers and three appended MTP layers, hidden width 4096, 64 query heads, four KV heads, head width 192, and a repeating pattern of three 512-token sliding layers followed by one sparse layer. There are 352 routed experts and eight active experts per token.

The reference is [StepFun's standalone implementation](https://huggingface.co/stepfun-ai/Step-5-Preview/tree/364b3b03fed2477aa0953fe855a2c4c8c48ee3cc/inference), revision `364b3b03fed2477aa0953fe855a2c4c8c48ee3cc`. The original preview GGUF can be used with restored indexer tensors; matching dimensions does not establish identical weights to the official release.

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

CPU/GPU kernels, cache lifecycle and MTP must implement this contract before CSA is enabled by default. Do not treat token-level top-4096 selection as equivalent to regional CSA.

## Numerical oracle

Run `python examples/step5/reference.py --output report.json --fixtures fixtures` in a Python environment with PyTorch and NumPy. This uses CPU calculations and does not load the model. It checks dense equivalence before the region budget is exceeded, causal selection, provider isolation, zero-centered norms and stable ties. Binary fixtures provide independent expected results for backend validation.

## Attention compatibility mode

Step-5 uses the reference attention scale by default. Set `LLAMA_STEP5_LEGACY_ATTENTION=1` to reproduce the previous `ssmax_s` scale and indexer normalization for a controlled dense comparison. This flag does not affect ordinary Step3.5 checkpoints. Indexer affine parameters, projections, normalization outputs and RoPE outputs are rounded to BF16 in the corrected path.

CPU normalization matched the independent PyTorch fixture exactly. CUDA on Ada and Ampere matched Q exactly; one of 32768 K values differed by one BF16 ULP (0.00390625), within the FP32 reduction tolerance. Existing GGML NORM, RMS_NORM and RoPE backend checks also passed on CUDA.

## Experimental grouped CSA

Set `LLAMA_STEP5_EXPERIMENTAL_CSA=1` to enable region compression and four independent provider groups. The older `LLAMA_STEP5_EXPERIMENTAL_SPARSE=1` still selects individual tokens and is a separate comparison mode. CSA requires F16 K/V and Flash Attention. Keep context shifting disabled.

The indexer uses BF16 projection/norm/RoPE boundaries and E4M3 score inputs. Precise CUDA trigonometry preserves the BF16 reference table at long positions despite the build's global fast-math flags. The attention kernel directly reads selected original K/V for each group and query, including multi-token verification batches.

Run `python examples/step5/check.py --libdir build-aliceai/bin --backend CUDA0 --output csa.json` with PyTorch and NumPy; `CPU` and other logical CUDA device numbers are also accepted. Checks cover the reference boundary lengths, signed head weights, stable ties, shuffled cache slots, independent sequences, and 192-wide attention with 64 query heads. Norm/RoPE fixtures at positions 0, 4096 and 143999 matched the official Triton kernel exactly on Ada and Ampere. Selected indices matched the independent oracle exactly in all fixtures. Rare FP32 summary reductions can straddle a BF16 midpoint before E4M3; the validator reports those elements and requires them to stay in the quantization interval of a 1e-6 FP32 error bound.

These numerical checks do not establish full-model long-context quality. CSA remains opt-in while cache, MTP and checkpoint quality are evaluated.

## Cache and memory

CSA stores canonical BF16 K/Z token projections in the indexer cache. Completed summaries are materialized separately from the current incomplete region and recomputed from that canonical cache. This costs extra cache memory but makes sequence removal, prefix copy and state serialization use the existing KV lifecycle; there is no hidden summary state to restore after speculative rejection. Future completed regions are excluded per query before selection.

The logical slot map handles independent sequences and physical cache fragmentation. Chunked prefill may append to an existing cache. Context shifting remains unsupported. CSA scratch is part of the GGML graph allocation, including the cuBLAS score workspace, so auto-fit measures it. The actual logical range bounds normal execution; full-cache graph reservation still measures the context limit.

Generate the small test GGUF with `PYTHONPATH=gguf-py python examples/step5/make-small.py`. Build `examples/step5/check-cache.cpp` against libllama and run it with the GGUF path and either `0` (separate streams) or `1` (unified cache). It checks chunk sizes 128/256, prefix copy, independent sequences, verification rollback, state restore into a context with microbatch 512, and CUDA graph reuse. Both modes kept the same greedy token; the maximum logit difference was 0.000273 on the deterministic four-layer model.

A six-GPU auto-fit plan was generated for 144000 context tokens and microbatch 512 with a 2048 MiB margin. This is memory planning, not yet a claim that 144K full-model quality passed.

## MTP verification

The existing three-head MTP implementation is reused, including its adaptive draft length. No new draft model is needed. Use `--spec-type draft-mtp --spec-draft-n-max 3`; set `LLAMA_MTP_ADAPTIVE=0` for fixed-budget comparisons. Disable drafting with a separate `--spec-type none --spec-draft-n-max 0` launch. Pass `--spec-type` once: repeated options append types.

`check-cache.cpp MODEL 0 verify` and `check-cache.cpp MODEL 1 verify` check all 12 acceptance counts for verification batches of two, three and four tokens. Each crosses an eight-token region boundary. After rejection, a clean target context that never decoded the rejected tokens is the reference. Both cache modes passed, with equal greedy tokens and maximum absolute logit error 0.000486 on the deterministic four-layer test model. The 0.001 tolerance covers batch-dependent arithmetic on this test model; it is not a bound for the full checkpoint.

The oracle and CPU/CUDA summaries also cover extreme Z values. Per-coordinate denominators use the reference clamp at 1e-20; an underflowing coordinate returns zero instead of NaN.

With the restored Q3_K_M weights and the same six-GPU placement, fixed MTP and target-only runs produced identical token IDs for the 7382-token and 15503-token retrieval prompts. All eight facts at 15K were correct. The 31028-token prompt returned both correct facts in both modes, but its requested explanation diverged after 46 common output tokens (93 MTP output tokens versus 91 target-only tokens). Repeating MTP on the same source version retained the divergence, so the summary denominator correction did not cause it. Replaying the common prefix selected the same next token in both modes. Full-model bit-exact greedy parity has not been established; batch-dependent arithmetic and index selection need further investigation. Keep CSA with MTP experimental.

Adaptive MTP was enabled and tested on the same 7K, 15K and 31K prompts. The 7K/15K outputs again matched target-only token IDs exactly. At 31K all requested facts remained correct, while the explanation diverged after the same 46-token prefix. The draft budget decreased: 74 draft tokens over 45 draft calls, compared with 102 tokens over 34 fixed-budget calls. The original 96-token output cap cut the adaptive explanation short; a cached repeat with a 256-token cap completed normally in 117 output tokens, with both facts correct. The test does not establish bit-exact greedy parity or an adaptive speed benefit.

## Full-model quality checks

The validation used the existing restored TypeSafeAI Q3_K_M checkpoint, not a newly converted official checkpoint. The model has eight GGUF splits and 1783 tensors. These results do not establish identity to StepFun's released weights or general long-context accuracy.

The complete inference configuration was tested on CUDA 13 with five 48 GiB Ada GPUs and one 24 GiB RTX 3090, plus CPU expert offload. Every GPU test used `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=2,1,0,5,4,3`. Numerical CPU operator checks also passed; CPU-only execution of the complete model and other GPU backends were not tested.

The long requests used context 144000 (allocated as 144128), batch 2048, microbatch 512, 20 CPU threads, F16 K/V, Flash Attention, and three MTP heads with existing draft-length adaptation. Context shifting was disabled. Auto-fit generated the placement with a 2048 MiB device margin; the saved placement was then fixed for paired runs. The main model buffers matched between target-only and MTP launches.

| Input tokens | Check | Result |
| --- | --- | --- |
| 7382 | Two archive facts | 2/2, identical output token IDs with target-only, fixed MTP and adaptive MTP |
| 15503 | Eight archive facts | 8/8, identical output token IDs in all three modes |
| 31028 | Two facts and a requested explanation | Both facts correct; target-only/MTP explanations differ after 46 common tokens |
| 59999 | Mixed archive, eight distributed facts | 8/8, normal end of response, no input truncation |
| 59999, cached | Same mixed archive | 8/8; 59995 cached input tokens, four tokens processed again |
| Approximately 60K | Follow-up asking for THETA and ALPHA in reverse order | 2/2; prefix reuse and normal end of response |
| 142980 | Mixed archive, eight distributed facts | 8/8, normal end of response, no input truncation |
| 142980, cached | Same mixed archive | 8/8; 142976 cached input tokens, four tokens processed again |
| Approximately 143K | Follow-up asking for THETA and ALPHA | 2/2; 143030 cached input tokens and 32 new prompt tokens |

The mixed archives contain Russian prose, Python, JSON, tables, request logs and English documents. `examples/step5/check-server.py` generates them using the running server's tokenizer, saves complete requests and responses, rejects contradictory values, and optionally checks prefix reuse and a follow-up turn. For example, run `python3 examples/step5/check-server.py --url http://127.0.0.1:8086 --tokens 60000 143000 --follow-up --output step5-quality` after the server is ready. Output budgets must fit in the selected context. The test deliberately disables sampling for repeatability.

The 31K adaptive explanation reached the original 96-token output limit; a cached repeat with a 256-token limit completed normally in 117 tokens, with both facts correct. This was an output-budget limit, not input truncation. Bit-exact full-model greedy parity has not been established beyond the shorter paired fixtures.

The server also passed a real function-call round trip: required tool selection, structured arguments, a tool-result message, and the final answer 17. Image checks used the separate projector on CPU: a synthetic red square and OCR of the rendered code 483726. A subsequent text-only request returned 56 with MTP drafting active again. Target-only and fixed-MTP launches both passed a tool-result round trip and the image checks. Target-only used string `tool_choice: "required"`; the fixed-MTP request used a named-object selector that the generic server treated as automatic selection. Named-object enforcement is not supported by this path. Parallel tool use, natural images and video are not covered by these fixtures.

These are functional quality checks. Performance comparisons must additionally control placement, warmup, cache reuse and repeated measurements; see the performance results for those conditions. The maximum checked initial prompt was 142980 tokens; the follow-up reached 143076 retained tokens. No 1M-context validation is claimed.
