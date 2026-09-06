# GLM5NEXT Local Changes Report

## Report scope

- Branch: `glm5next-upstream-optimized`
- Comparison upstream: `origin/glm5next/upstream`
- Merge base: `f30bed88717059d8a4728864c88f8abad8d329a0`
- Implementation commits on top of upstream: 31
- Order below: chronological, from the first change to the latest

Documentation-only commits that change this report and its Russian version
are excluded from the implementation list.

The list can be reproduced with:

```bash
git log --reverse --oneline origin/glm5next/upstream..HEAD -- . \
  ':(exclude)GLM5NEXT_LOCAL_CHANGES.md' \
  ':(exclude)GLM5NEXT_LOCAL_CHANGES_EN.md'
```

## Summary

The work was divided into several areas:

1. Reduce the cost of sparse attention at long context lengths.
2. Add a persistent cache for completed indexer pool keys.
3. Add complete MTP support, including prompt reuse, multimodality, and multiple slots.
4. Eliminate CPU copies of hidden states and embeddings.
5. Fix auto-fit, LCP rewind, multi-slot rebuild, and incremental update sizing.
6. Optimize indexed FlashAttention, Lightning Indexer, and CUDA Graph execution.
7. Optimize MoE decode and fused Q3_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS expert down reduction.

## Complete commit list

### 1. `b888281` - `glm5next: compact sparse decode attention`

- Change: decode attention now operates on a compact set containing the selected pool members, the incomplete tail, and alignment padding instead of the full KV range.
- Purpose: limit attention work to the selected tokens and reduce generation slowdown as the context grows.

### 2. `7588579` - `glm5next: cache completed pool keys`

- Change: added persistent storage for compressed keys of completed pools and incremental updates for only new or invalidated entries.
- Purpose: avoid recomputing all completed pool keys on every decode step, so compression cost no longer grows with the entire context.

### 3. `f947340` - `glm5next: add full MTP decoding`

- Change: added a complete MTP draft context, hidden-state transfer, MTP KV/recurrent/pool cache management, and speculative decoding integration.
- Purpose: use the built-in next-token head of GLM-5.3-Flash for speculative decoding without a separate draft model.

### 4. `fb88d4a` - `glm5next: skip empty recurrent MTP cache`

- Change: recurrent MTP cache operations are skipped when a sequence has no state yet.
- Purpose: avoid access to an empty cache and failures on new or reset MTP sequences.

### 5. `e467c0b` - `glm5next: handle empty recurrent MTP graph input`

- Change: the recurrent MTP graph input now handles a missing input state correctly.
- Purpose: complement the previous cache guard at graph level and prevent graph construction or execution failures with an empty MTP state.

### 6. `3e789bb` - `fit: track shared MTP target placement`

- Change: auto-fit now accounts for target placement and the built-in nextn/MTP layers when estimating the complete model and context allocation.
- Purpose: prevent an incorrect free-VRAM estimate followed by OOM while creating an MTP context at a large context length.

### 7. `4edb38e` - `cuda: accelerate GLM lightning top-k selection`

- Change: added a specialized CUDA top-k path for the GLM Lightning Indexer data shape.
- Purpose: reduce pool-selection time, which became a significant part of decode at long context lengths.

### 8. `aa567f5` - `server: keep MTP state aligned across prompt reuse`

- Change: target, draft, and speculative states are synchronized during LCP reuse, rewind, and prompt reprocessing; a safe full-reprocess fallback was added.
- Purpose: eliminate non-consecutive position warnings, draft/target position divergence, and `ctx_dft cannot catch up` failures.

### 9. `26cf4fe` - `server: suspend MTP for multimodal requests`

- Change: MTP was temporarily disabled for requests containing media embeddings.
- Purpose: preserve correctness until complete synchronization of the multimodal target and draft paths was available.
- Status: this safety restriction was later replaced by full support in `cfcdf4e`.

### 10. `7ac7124` - `glm5next: accelerate long-context sparse prefill`

- Change: added indexed FlashAttention that reads K/V directly through selected indices, together with CPU/CUDA index infrastructure and dense/indexed path selection.
- Purpose: avoid constructing and processing the complete dense attention range for every prefill token when the KV cache is large.

### 11. `fcd2441` - `glm5next: keep small MTP batches off indexed attention`

- Change: small speculative batches remain on the dense attention path; indexed attention is enabled only when the batch/KV size provides enough parallel work.
- Purpose: avoid a performance cliff at long context lengths when the indexed kernel is underutilized by a small MTP batch.

### 12. `cfcdf4e` - `glm5next: enable multimodal MTP fast path`

- Change: synchronized MTP with media batches, transferred vision embeddings directly between CUDA backends, and cached compatible vision graphs.
- Purpose: retain MTP generation acceleration after multimodal prefill and remove unnecessary GPU-CPU-GPU copies and repeated vision graph construction.

### 13. `1315090` - `glm5next: accelerate long-context MTP attention`

- Change: parallelized indexed attention across multiple warps; added reuse of indexer/pool state between related MTP iterations and compatibility checks for graph inputs.
- Purpose: accelerate small MTP batches on a large KV cache, where single-warp execution and repeated index preparation were bottlenecks.

### 14. `99dc6b4` - `glm5next: align metadata and quantization handling`

- Change: synchronized GGUF metadata keys, tokenizer fallback/prefix handling, the MTP indexer-sharing flag, the vision SwiGLU clamp key, and quantization rules.
- Purpose: convert and load new GLM-5.3 files correctly without losing architecture parameters or quantizing incompatible tensors.

### 15. `4a7b87f` - `perf(glm5): preserve MTP pool cache across draft steps`

- Change: added Q8_0 indexed CUDA attention; completed pool keys persist between MTP steps, while invalidation is limited to affected speculative blocks.
- Purpose: avoid discarding useful persistent cache data after every draft/verify cycle and accelerate indexed attention with a Q8 KV cache.

### 16. `1f0fccd` - `fix(glm5): size pool updates for pending dirty keys`

- Change: incremental pool-update capacity now accounts for all pending dirty keys, including changes produced by MTP and LCP rewind.
- Purpose: prevent update-capacity overflow, incorrect graph reuse, and crashes after speculative-state rollback.

### 17. `40efd56` - `cuda: optimize GLM sparse attention and MoE decode`

- Change: ported and adapted fused SwiGLU clamp, multi-token MoE kernels, weighted expert reduction, FlashAttention swizzling, and multi-GPU graph optimization; adapted sparse MMA attention to the GLM compact-index path.
- Purpose: obtain the improvements from the newer CUDA/MoE implementation while retaining compatibility with the local sparse-attention and multi-GPU design.

### 18. `155c6a1` - `cuda: avoid weighted MoE fusion for single-token decode`

- Change: retained the general fused weighted-expert reduction for prefill and verification batches, but disabled it for ordinary single-token decode.
- Purpose: measurements showed that the existing vector reduction path is faster than the general multi-token fusion for one token.

### 19. `7e9c947` - `speculative : keep MTP hidden states on backend`

- Change: MTP hidden states remain in backend/GPU buffers and are passed to the next MTP graph without a mandatory CPU copy.
- Purpose: eliminate synchronization and PCIe transfer on every speculative step; the initial GPU-direct path targeted a single slot.

### 20. `01c44d7` - `speculative: support GPU-direct MTP with multiple slots`

- Change: added separate backend hidden-state regions for each sequence and graph-side gathering of non-contiguous rows for multiple `seq_id` values.
- Purpose: retain GPU-direct MTP with `n_slots > 1` without mixing the state of parallel requests.

### 21. `1ffe9e4` - `cuda: vectorize indexed flash attention loads`

- Change: adjacent F16 and Q8_0 K/V elements are loaded in groups of four; Q8_0 uses separately tuned launch geometry.
- Purpose: reduce instruction count and improve the efficiency of random reads in `fattn-indexed`.

### 22. `469a0d9` - `cuda: tile Lightning Indexer prefill scoring`

- Change: FP32 pool keys are reused by a group of prefill queries inside a tiled CUDA kernel; the decode path remains unchanged.
- Purpose: reduce repeated reads of the same pool keys during prefill and mitigate indexer slowdown as the number of pools grows.

### 23. `bc38ed4` - `cuda: reuse temporal hints for GLM pool top-k`

- Change: previous top-k indices are used as a temporal hint to estimate a threshold and build a smaller exact candidate set; an exact full-selection fallback remains available when the hint is unsuitable.
- Purpose: exploit similarity between adjacent decode states and reduce exact top-k cost without changing the selected result.

### 24. `fa1eb32` - `speculative: adapt MTP draft length`

- Change: MTP draft length changes automatically from per-sequence acceptance history, subject to `n_min`, `n_max`, `p_min`, and remaining context space.
- Purpose: avoid wasting verification compute on an overly long draft when acceptance is low, while increasing the draft length after consistently complete acceptance.

### 25. `de0e283` - `cuda: cache graphs by topology`

- Change: the CUDA Graph cache is keyed by a hash of node topology, shapes, strides, and parameters; a bounded LRU cache stores up to 64 variants.
- Purpose: reuse CUDA Graphs for previously seen batch/shape variants instead of repeatedly recapturing graphs during MTP and changing prompt sizes.

### 26. `eb905c3` - `cuda: fuse GLM pool index expansion`

- Change: selected-pool gathering, validity expansion, tail append, and compact concatenation were combined into one CPU/CUDA `KPOOL_EXPAND` operation; invalid entries are passed as negative indices directly to mask-free indexed attention.
- Purpose: reduce graph-node count, temporary tensors, and memory traffic before sparse attention. The faster materialized path remains in use for short decode.

### 27. `3589bab` - `cuda: fuse GLM expert down reduction`

- Change: combined the single-token Q5_K expert-down `MUL_MAT_ID` and subsequent routing-weight multiplication/reduction into one CUDA kernel for the GLM-5.3-Flash `2048x4096` shape.
- Purpose: avoid materializing eight intermediate expert outputs and eliminate separate MUL/view/add kernels during decode.
- A/B control: set `GGML_CUDA_MOE_DOWN_REDUCE=0` to disable the fusion.

### 28. `4d8feda` - `cuda: extend GLM down reduction fusion to Q6_K`

- Change: generalized the fused down/reduction kernel from Q5_K to Q6_K and added correctness/performance tests for both formats.
- Purpose: enable the optimization for `UD-Q5_K_XL`, whose expert gate/up tensors use Q5_K while `ffn_down_exps.weight` is actually stored as Q6_K.

### 29. `549b1bb` - `cuda: generalize MoE down reduction fusion`

- Change: removed the restriction to the single `2048x4096` shape; expanded the tested range to `n_ff=768..2048` and `n_embd=2048..7168` for Q5_K/Q6_K, top-8, single-token decode.
- Purpose: apply the same fast path to compatible Qwen-like, GLM, and DeepSeek-like MoE shapes. The standard GLM-5.3 expert-down shape of `2048x6144` is also within this range.

### 30. `8ccb84f` - `fix(glm5): track pool cache rebuilds per stream`

- Change: replaced the global persistent pool-key cache rebuild flag with per-physical-KV-stream state; graph build/reuse and rebuild completion now operate only on the stream range actually processed. Added a two-slot checkpoint-restore regression test.
- Purpose: restoring a checkpoint for one slot clears the pool maps of every stream. Previously, rebuilding the first slot incorrectly marked the entire cache ready, after which the second slot entered the incremental path with insufficient capacity and failed with `incremental pool-key update capacity is too small`.
- Validation: the CUDA Release build passed; cached and non-cached logits matched with `max abs = 0`; the new multi-stream restore test completed with status `ok`.

### 31. `d675ef7` - `cuda: extend fused MoE down reduction to low-bit quants`

- Change: extended the single-token fused expert-down and weighted-reduction path from Q5_K/Q6_K to Q3_K, IQ3_XXS, and IQ4_XS. The test matrix covers `768x2048`, `2048x4096`, `2048x6144`, and `2048x7168` shapes for all five types.
- Purpose: use the specialized decode path for `UD-Q3_K_XL`, `UD-IQ3_XXS`, and `UD-IQ4_XS`, including the expert-down shapes used by GLM-5.3-Flash and standard GLM-5.3.
- Q3_K A/B: the fused kernel is 43-45% faster than the unfused chain for GLM shapes on RTX 4090 and 34-35% faster on RTX 3090.
- IQ3_XXS A/B: the gain is 35-38% on RTX 4090 and 31-33% on RTX 3090; for IQ4_XS it is 27-39% and 14-18%, respectively.
- Validation: the complete CUDA correctness regression passed `20/20`; dedicated Q3_K and IQ3_XXS checks passed on both RTX 4090 and RTX 3090; `llama-server` built successfully.

## Important dependencies between changes

- `26cf4fe` was a temporary correctness fallback; full multimodal MTP support was added in `cfcdf4e`.
- `155c6a1` narrows the general MoE fusion introduced by `40efd56`: single-token decode uses the faster specialized path.
- `7e9c947` removed the CPU round trip for one MTP stream, and `01c44d7` extended GPU-direct state storage to multiple slots.
- `3589bab`, `4d8feda`, `549b1bb`, and `d675ef7` are consecutive stages of one optimization: the Q5_K GLM shape, then Q6_K, other tested MoE shapes, and the low-bit Q3_K/IQ3_XXS/IQ4_XS formats.
- `7588579`, `4a7b87f`, `1f0fccd`, and `8ccb84f` together form the persistent pool-key cache with selective invalidation, sufficient update capacity, and independent rebuild state for multiple slots.

## Diagnostic controls

- `GGML_CUDA_MOE_DOWN_REDUCE=0` disables fused expert-down reduction for A/B testing.
- `LLAMA_GLM5_INDEXED_ATTN=0` keeps the dense sparse-attention path; value `2` forces the indexed path where supported.
- `LLAMA_MTP_ADAPTIVE=0` disables adaptive MTP draft length.
- `GGML_CUDA_GRAPH_SHAPE_CACHE=0` restores the previous CUDA Graph cache key strategy.

## Current state

- Latest implementation commit: `d675ef7`
- All listed changes are present in the history of `glm5next-upstream-optimized`.
- This report was generated from the actual `origin/glm5next/upstream..HEAD` range, excluding documentation-only commits from the list.
