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
