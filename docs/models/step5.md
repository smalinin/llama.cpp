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

The implementation below follows this contract. Token-level top-4096 selection is a separate legacy experiment and is not equivalent to regional CSA.

## Numerical oracle

Run `python examples/step5/reference.py --output report.json --fixtures fixtures` in a Python environment with PyTorch and NumPy. This uses CPU calculations and does not load the model. It checks dense equivalence before the region budget is exceeded, causal selection, provider isolation, zero-centered norms and stable ties. Binary fixtures provide independent expected results for backend validation.

## Attention compatibility mode

Step-5 uses the reference attention scale by default. Set `LLAMA_STEP5_LEGACY_ATTENTION=1` to reproduce the previous `ssmax_s` scale and indexer normalization for a controlled dense comparison. This flag does not affect ordinary Step3.5 checkpoints. Indexer affine parameters, projections, normalization outputs and RoPE outputs are rounded to BF16 in the corrected path.

CPU normalization matched the independent PyTorch fixture exactly. CUDA on Ada and Ampere matched Q exactly; one of 32768 K values differed by one BF16 ULP (0.00390625), within the FP32 reduction tolerance. Existing GGML NORM, RMS_NORM and RoPE backend checks also passed on CUDA.

## Experimental grouped CSA

Set `LLAMA_STEP5_EXPERIMENTAL_CSA=1` to enable region compression and four independent provider groups. The older `LLAMA_STEP5_EXPERIMENTAL_SPARSE=1` still selects individual tokens and is a separate comparison mode. CSA requires F16 K/V and Flash Attention. Keep context shifting disabled.

The indexer uses BF16 projection/norm/RoPE boundaries and E4M3 score inputs. Precise CUDA trigonometry preserves the BF16 reference table at long positions despite the build's global fast-math flags. The attention path maps selected logical positions to physical rows and reads original K/V through native indexed Flash Attention, including multi-token verification batches. Unusual layouts retain a direct attention kernel fallback. Independent logical sequences in a unified physical cache use the indexed path.

Run `python examples/step5/check.py --libdir build-aliceai/bin --backend CUDA0 --output csa.json` with PyTorch and NumPy; `CPU` and other logical CUDA device numbers are also accepted. Checks cover the reference boundary lengths, signed head weights, stable ties, shuffled cache slots, independent sequences, and 192-wide attention with 64 query heads. Norm/RoPE fixtures at positions 0, 4096 and 143999 matched the official Triton kernel exactly on Ada and Ampere. Selected indices matched the independent oracle exactly in all fixtures. Rare FP32 summary reductions can straddle a BF16 midpoint before E4M3; the validator reports those elements and requires them to stay in the quantization interval of a 1e-6 FP32 error bound.

The accelerated path also covers queries at positions 0, 7 and 8 with a populated 8192-slot cache. Native CUDA Flash Attention uses F16 probability and accumulator tiles; it is not an FP32-accumulation promise. Indexed attention requires NMSE below 1e-6 and a value-scaled absolute bound; the one-key case is checked separately. CPU indexed attention honors the explicit FP32 accumulator request. Numerical operator checks and full-model checks are separate.

## Cache and memory

CSA stores canonical BF16 K/Z token projections in the indexer cache. Completed summaries are materialized separately from the current incomplete region and recomputed from that canonical cache. This costs extra cache memory but makes sequence removal, prefix copy and state serialization use the existing KV lifecycle; there is no hidden summary state to restore after speculative rejection. Future completed regions are excluded per query before selection.

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

The maximum checked initial prompt was 142980 tokens; the earlier adaptive-MTP follow-up retained 143076 tokens. No 1M-context validation is claimed. Performance comparisons control warmup, placement and prompt-cache reuse separately; see [the CUDA performance report](step5-performance.md).
