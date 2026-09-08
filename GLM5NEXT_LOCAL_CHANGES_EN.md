# GLM5NEXT Local Changes Report

## Report scope

- Repository: `/home/sergei/Github/llama.cpp`
- Branch: `my_build_glm53_flash`
- Target baseline before the port: `465e49b9c`
- Imported GLM5NEXT foundation commits: 32
- Local implementation and integration commits described below: 37
- Order below: chronological, from the first change to the latest

Documentation-only commits are excluded from the implementation list. The
original patch was based on Unsloth's
[`glm5next/upstream`](https://github.com/unslothai/llama.cpp/tree/glm5next/upstream)
branch. The optimization series was developed on top of that foundation in
`glm5next-upstream-optimized`, then ported and adapted to the newer APIs present
in this target branch.

The list can be reproduced with:

```bash
git log --reverse --oneline 98195add5..HEAD -- . \
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
7. Optimize MoE decode and fused Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS expert down reduction.

## Local optimization and integration commit list

### 1. `fe686c01b` - `glm5next: compact sparse decode attention`

- Change: decode attention now operates on a compact set containing the selected pool members, the incomplete tail, and alignment padding instead of the full KV range.
- Purpose: limit attention work to the selected tokens and reduce generation slowdown as the context grows.

### 2. `6ba576144` - `glm5next: cache completed pool keys`

- Change: added persistent storage for compressed keys of completed pools and incremental updates for only new or invalidated entries.
- Purpose: avoid recomputing all completed pool keys on every decode step, so compression cost no longer grows with the entire context.

### 3. `7eb2a1a93` - `glm5next: add full MTP decoding`

- Change: added a complete MTP draft context, hidden-state transfer, MTP KV/recurrent/pool cache management, and speculative decoding integration.
- Purpose: use the built-in next-token head of GLM-5.3-Flash for speculative decoding without a separate draft model.

### 4. `e82c72ebf` - `glm5next: skip empty recurrent MTP cache`

- Change: recurrent MTP cache operations are skipped when a sequence has no state yet.
- Purpose: avoid access to an empty cache and failures on new or reset MTP sequences.

### 5. `d64037d10` - `glm5next: handle empty recurrent MTP graph input`

- Change: the recurrent MTP graph input now handles a missing input state correctly.
- Purpose: complement the previous cache guard at graph level and prevent graph construction or execution failures with an empty MTP state.

### 6. `38bebaf8e` - `fit: track shared MTP target placement`

- Change: auto-fit now accounts for target placement and the built-in nextn/MTP layers when estimating the complete model and context allocation.
- Purpose: prevent an incorrect free-VRAM estimate followed by OOM while creating an MTP context at a large context length.

### 7. `9abae4baa` - `cuda: accelerate GLM lightning top-k selection`

- Change: added a specialized CUDA top-k path for the GLM Lightning Indexer data shape.
- Purpose: reduce pool-selection time, which became a significant part of decode at long context lengths.

### 8. `ccdf0ae2c` - `server: keep MTP state aligned across prompt reuse`

- Change: target, draft, and speculative states are synchronized during LCP reuse, rewind, and prompt reprocessing; a safe full-reprocess fallback was added.
- Purpose: eliminate non-consecutive position warnings, draft/target position divergence, and `ctx_dft cannot catch up` failures.

### 9. `08762307d` - `server: suspend MTP for multimodal requests`

- Change: MTP was temporarily disabled for requests containing media embeddings.
- Purpose: preserve correctness until complete synchronization of the multimodal target and draft paths was available.
- Status: this safety restriction was later replaced by full support in `432d41f03`.

### 10. `fddd7101e` - `glm5next: accelerate long-context sparse prefill`

- Change: added indexed FlashAttention that reads K/V directly through selected indices, together with CPU/CUDA index infrastructure and dense/indexed path selection.
- Purpose: avoid constructing and processing the complete dense attention range for every prefill token when the KV cache is large.

### 11. `53350fd8e` - `glm5next: keep small MTP batches off indexed attention`

- Change: small speculative batches remain on the dense attention path; indexed attention is enabled only when the batch/KV size provides enough parallel work.
- Purpose: avoid a performance cliff at long context lengths when the indexed kernel is underutilized by a small MTP batch.

### 12. `432d41f03` - `glm5next: enable multimodal MTP fast path`

- Change: synchronized MTP with media batches, transferred vision embeddings directly between CUDA backends, and cached compatible vision graphs.
- Purpose: retain MTP generation acceleration after multimodal prefill and remove unnecessary GPU-CPU-GPU copies and repeated vision graph construction.

### 13. `22a4a910d` - `glm5next: accelerate long-context MTP attention`

- Change: parallelized indexed attention across multiple warps; added reuse of indexer/pool state between related MTP iterations and compatibility checks for graph inputs.
- Purpose: accelerate small MTP batches on a large KV cache, where single-warp execution and repeated index preparation were bottlenecks.

### 14. `f33ad5c16` - `glm5next: align metadata and quantization handling`

- Change: synchronized GGUF metadata keys, tokenizer fallback/prefix handling, the MTP indexer-sharing flag, the vision SwiGLU clamp key, and quantization rules.
- Purpose: convert and load new GLM-5.3 files correctly without losing architecture parameters or quantizing incompatible tensors.

### 15. `9826966a5` - `perf(glm5): preserve MTP pool cache across draft steps`

- Change: added Q8_0 indexed CUDA attention; completed pool keys persist between MTP steps, while invalidation is limited to affected speculative blocks.
- Purpose: avoid discarding useful persistent cache data after every draft/verify cycle and accelerate indexed attention with a Q8 KV cache.

### 16. `16699cd8a` - `fix(glm5): size pool updates for pending dirty keys`

- Change: incremental pool-update capacity now accounts for all pending dirty keys, including changes produced by MTP and LCP rewind.
- Purpose: prevent update-capacity overflow, incorrect graph reuse, and crashes after speculative-state rollback.

### 17. `4e3e75392` - `cuda: optimize GLM sparse attention and MoE decode`

- Change: ported and adapted fused SwiGLU clamp, multi-token MoE kernels, weighted expert reduction, FlashAttention swizzling, and multi-GPU graph optimization; adapted sparse MMA attention to the GLM compact-index path.
- Purpose: obtain the improvements from the newer CUDA/MoE implementation while retaining compatibility with the local sparse-attention and multi-GPU design.

### 18. `171e93b6c` - `cuda: avoid weighted MoE fusion for single-token decode`

- Change: retained the general fused weighted-expert reduction for prefill and verification batches, but disabled it for ordinary single-token decode.
- Purpose: measurements showed that the existing vector reduction path is faster than the general multi-token fusion for one token.

### 19. `7328ef9ae` - `speculative : keep MTP hidden states on backend`

- Change: MTP hidden states remain in backend/GPU buffers and are passed to the next MTP graph without a mandatory CPU copy.
- Purpose: eliminate synchronization and PCIe transfer on every speculative step; the initial GPU-direct path targeted a single slot.

### 20. `874703118` - `speculative: support GPU-direct MTP with multiple slots`

- Change: added separate backend hidden-state regions for each sequence and graph-side gathering of non-contiguous rows for multiple `seq_id` values.
- Purpose: retain GPU-direct MTP with `n_slots > 1` without mixing the state of parallel requests.

### 21. `7a147b345` - `cuda: vectorize indexed flash attention loads`

- Change: adjacent F16 and Q8_0 K/V elements are loaded in groups of four; Q8_0 uses separately tuned launch geometry.
- Purpose: reduce instruction count and improve the efficiency of random reads in `fattn-indexed`.

### 22. `18ce13822` - `cuda: tile Lightning Indexer prefill scoring`

- Change: FP32 pool keys are reused by a group of prefill queries inside a tiled CUDA kernel; the decode path remains unchanged.
- Purpose: reduce repeated reads of the same pool keys during prefill and mitigate indexer slowdown as the number of pools grows.

### 23. `9ed8aaec5` - `cuda: reuse temporal hints for GLM pool top-k`

- Change: previous top-k indices are used as a temporal hint to estimate a threshold and build a smaller exact candidate set; an exact full-selection fallback remains available when the hint is unsuitable.
- Purpose: exploit similarity between adjacent decode states and reduce exact top-k cost without changing the selected result.

### 24. `bf88c19a9` - `speculative: adapt MTP draft length`

- Change: MTP draft length changes automatically from per-sequence acceptance history, subject to `n_min`, `n_max`, `p_min`, and remaining context space.
- Purpose: avoid wasting verification compute on an overly long draft when acceptance is low, while increasing the draft length after consistently complete acceptance.

### 25. `eb97e64fd` - `cuda: cache graphs by topology`

- Change: the CUDA Graph cache is keyed by a hash of node topology, shapes, strides, and parameters; a bounded LRU cache stores up to 64 variants.
- Purpose: reuse CUDA Graphs for previously seen batch/shape variants instead of repeatedly recapturing graphs during MTP and changing prompt sizes.

### 26. `c7a5fd3ef` - `cuda: fuse GLM pool index expansion`

- Change: selected-pool gathering, validity expansion, tail append, and compact concatenation were combined into one CPU/CUDA `KPOOL_EXPAND` operation; invalid entries are passed as negative indices directly to mask-free indexed attention.
- Purpose: reduce graph-node count, temporary tensors, and memory traffic before sparse attention. The faster materialized path remains in use for short decode.

### 27. `11ce5487a` - `cuda: fuse GLM expert down reduction`

- Change: combined the single-token Q5_K expert-down `MUL_MAT_ID` and subsequent routing-weight multiplication/reduction into one CUDA kernel for the GLM-5.3-Flash `2048x4096` shape.
- Purpose: avoid materializing eight intermediate expert outputs and eliminate separate MUL/view/add kernels during decode.
- A/B control: set `GGML_CUDA_MOE_DOWN_REDUCE=0` to disable the fusion.

### 28. `f0a3bd2df` - `cuda: extend GLM down reduction fusion to Q6_K`

- Change: generalized the fused down/reduction kernel from Q5_K to Q6_K and added correctness/performance tests for both formats.
- Purpose: enable the optimization for `UD-Q5_K_XL`, whose expert gate/up tensors use Q5_K while `ffn_down_exps.weight` is actually stored as Q6_K.

### 29. `41c20e310` - `cuda: generalize MoE down reduction fusion`

- Change: removed the restriction to the single `2048x4096` shape; expanded the tested range to `n_ff=768..2048` and `n_embd=2048..7168` for Q5_K/Q6_K, top-8, single-token decode.
- Purpose: apply the same fast path to compatible Qwen-like, GLM, and DeepSeek-like MoE shapes. The standard GLM-5.3 expert-down shape of `2048x6144` is also within this range.

### 30. `994bc1d31` - `fix(glm5): track pool cache rebuilds per stream`

- Change: replaced the global persistent pool-key cache rebuild flag with per-physical-KV-stream state; graph build/reuse and rebuild completion now operate only on the stream range actually processed. Added a two-slot checkpoint-restore regression test.
- Purpose: restoring a checkpoint for one slot clears the pool maps of every stream. Previously, rebuilding the first slot incorrectly marked the entire cache ready, after which the second slot entered the incremental path with insufficient capacity and failed with `incremental pool-key update capacity is too small`.
- Validation: the CUDA Release build passed; cached and non-cached logits matched with `max abs = 0`; the new multi-stream restore test completed with status `ok`.

### 31. `7dec6d343` - `cuda: extend fused MoE down reduction to low-bit quants`

- Change: extended the single-token fused expert-down and weighted-reduction path from Q5_K/Q6_K to Q3_K, IQ3_XXS, and IQ4_XS. The test matrix covers `768x2048`, `2048x4096`, `2048x6144`, and `2048x7168` shapes for all five types.
- Purpose: use the specialized decode path for `UD-Q3_K_XL`, `UD-IQ3_XXS`, and `UD-IQ4_XS`, including the expert-down shapes used by GLM-5.3-Flash and standard GLM-5.3.
- Q3_K A/B: the fused kernel is 43-45% faster than the unfused chain for GLM shapes on RTX 4090 and 34-35% faster on RTX 3090.
- IQ3_XXS A/B: the gain is 35-38% on RTX 4090 and 31-33% on RTX 3090; for IQ4_XS it is 27-39% and 14-18%, respectively.
- Validation: the complete CUDA correctness regression passed `20/20`; dedicated Q3_K and IQ3_XXS checks passed on both RTX 4090 and RTX 3090; `llama-server` built successfully.

### 32. `063932028` - `perf(glm5): keep short prefill on dense attention`

- Change: automatic indexed-attention selection no longer switches merely because the prefill batch contains at least 4096 tokens; it waits until the KV cache reaches 32768 tokens. Force-on mode remains available for A/B testing.
- Purpose: preserve the faster contiguous dense FlashAttention path at short context lengths, where indirect indexed reads and extra graph nodes cost more than they save.

### 33. `cdbd7de4b` - `cuda: add Q4_K fused MoE down reduction`

- Change: extended the single-token fused expert-down and weighted-reduction CUDA kernel to `Q4_K`, and added it to the correctness and performance test matrices.
- Purpose: enable the specialized MoE decode path for regular `Q4_K` expert-down tensors, including compatible `Q4_K_M` model layouts, instead of falling back to the generic materialize-and-reduce chain.
- Validation: the complete fused MoE CUDA matrix passed `24/24` across Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, and IQ3_XXS.

### 34. `6fc4fd2dc` - `glm5next: adapt port to current APIs`

- Change: adapted the port to the newer target APIs by passing the compact KV width to `build_attn_mha`, using per-layer expert FFN widths and expert counts, and matching the current const-qualified multimodal image-preprocessor interface.
- Purpose: preserve the intended sparse-attention, MoE, and multimodal behavior while making the complete GLM5NEXT series build and run correctly on the newer target branch.

### 35. `440637026` - `tests: cover GLM5NEXT sequential MTP and multi-stream indexed attention`

- Change: added a synthetic GLM5NEXT model with a real NextN/MTP block and exercised two sequential draft steps, including selection capture/reuse and CPU/CUDA logit comparison. Extended indexed FlashAttention tests to two independent streams with distinct valid KV ranges, padded/invalid indices, masked and mask-free paths, and F16/Q8_0 KV types.
- Purpose: protect MTP indexer reuse and per-slot indirect KV addressing before further changes remove redundant indexer work and host synchronization from the draft loop.
- Validation: the GLM5NEXT architecture test passed on CPU and all six CUDA devices; indexed FlashAttention passed `11/11` on SM89 and SM86; the four new multi-stream cases passed compute-sanitizer with zero errors.

### 36. `69b0498ef` - `glm5next: skip redundant MTP indexer key updates`

- Change: after the first MTP draft iteration has stored its shared sparse-attention selection, later iterations return that backend-resident selection before constructing the indexer K projection, compressor gate, K/G packing, and cache write. The host graph input now tolerates the deliberately pruned K/G destination while retaining tail-map refresh and dirty-pool tracking for normal target catch-up.
- Purpose: remove indexer work whose results cannot be consumed by the remaining draft iterations; accepted positions are written by target catch-up and rejected positions are discarded.
- Correctness: the synthetic MTP regression compares every greedy draft decision with `LLAMA_GLM5_MTP_TOPK_SHARE=0`, asserts that the initial iteration still computes K/G and pool scores, and asserts that subsequent reuse graphs contain none of those operations. Equal greedy candidates imply equal greedy target acceptance decisions for the same target logits.
- Validation: the CUDA Release targets built successfully; the GLM5NEXT architecture regression passed on CPU and all six CUDA devices; the RTX 4090 compute-sanitizer run reported zero errors. The real GLM-5.3-Flash pool-cache test produced identical cached/uncached logits (`max abs = 0`), identical argmax, and a successful two-stream restore.

### 37. `b297df10f` - `glm5next: keep MTP draft iterations on backend`

- Change: added persistent per-sequence device buffers for the sampled token, its top-10 confidence, and the next hidden-state row. After the seed iteration, the GLM5NEXT MTP graph gathers these inputs directly on the backend and writes the next iteration's state without copying logits or hidden states through the CPU. Only the compact final token/probability arrays are read after the complete draft run.
- Scope and fallback: the fast path supports one or more independent slots, adaptive and request-specific draft limits, `n_min`, `p_min`, and remaining-context limits. It is selected only for a single GLM5NEXT NextN head with GPU-resident hidden state and a supported backend top-k sampler; chain-head/shared-memory configurations and unsupported samplers retain the host loop. Set `LLAMA_MTP_DEVICE_DRAFT=0` to force that fallback.
- Correctness: synthetic two-sequence tests compare every device-selected token and top-10 softmax probability with the host reference, verify seed indexer evaluation and later indexer reuse, and exercise the fallback switch. The architecture regression passed on CPU and all six CUDA devices, and RTX 4090 compute-sanitizer reported zero errors. The sparse/indexed-attention regression and the real GGUF pool-cache/multi-stream restore test also passed.
- End-to-end validation: two simultaneous server slots completed 384-token requests without state overlap or position errors. With `n_max=3`, `p_min=0`, adaptive length disabled, and identical greedy output/acceptance, the median short-context generation rate was 49.99 versus 49.09 tokens/s for the host fallback (+1.8%; three 256-token runs). At 55,000 prompt tokens, it was 60.45 versus 59.69 tokens/s (+1.27%; one 256-token run), with identical output and the same 185/207 accepted/drafted tokens.

## Important dependencies between changes

- `08762307d` was a temporary correctness fallback; full multimodal MTP support was added in `432d41f03`.
- `171e93b6c` narrows the general MoE fusion introduced by `4e3e75392`: single-token decode uses the faster specialized path.
- `7328ef9ae` removed the CPU round trip for one MTP stream, and `874703118` extended GPU-direct state storage to multiple slots.
- `11ce5487a`, `f0a3bd2df`, `41c20e310`, `7dec6d343`, and `cdbd7de4b` are consecutive stages of the fused MoE expert-down optimization: the Q5_K GLM shape, Q6_K, other tested MoE shapes, low-bit Q3_K/IQ3_XXS/IQ4_XS, and finally Q4_K.
- `6ba576144`, `9826966a5`, `16699cd8a`, and `994bc1d31` together form the persistent pool-key cache with selective invalidation, sufficient update capacity, and independent rebuild state for multiple slots.
- `158253632` is a pre-existing target-branch auto-fit prerequisite retained during the port; `38bebaf8e` adds the related shared-MTP placement accounting from the local series.

## Diagnostic controls

- `GGML_CUDA_MOE_DOWN_REDUCE=0` disables fused expert-down reduction for A/B testing.
- `LLAMA_GLM5_INDEXED_ATTN=0` keeps the dense sparse-attention path; value `2` forces the indexed path where supported.
- `LLAMA_GLM5_POOL_CACHE=0` disables the persistent completed-pool-key cache.
- `LLAMA_GLM5_KPOOL_EXPAND=0` disables fused pool-index expansion.
- `LLAMA_GLM5_MTP_TOPK_SHARE=0` disables MTP index-selection reuse between draft iterations.
- `GGML_CUDA_TOPK_TEMPORAL=0` disables temporal top-k hints.
- `GGML_CUDA_TOPK_RADIX_SELECT=0` disables the CUDA radix-selection top-k path.
- `LLAMA_MTP_ADAPTIVE=0` disables adaptive MTP draft length.
- `LLAMA_MTP_DEVICE_DRAFT=0` disables the backend-resident GLM5NEXT MTP inner loop and restores per-step host sampling/readback.
- `GGML_CUDA_GRAPH_SHAPE_CACHE=0` restores the previous CUDA Graph cache key strategy.

## Current state

- Latest implementation/integration commit: `b297df10f`
- All listed changes are present on `my_build_glm53_flash` in `/home/sergei/Github/llama.cpp`.
- The complete CUDA Release build succeeds in `build-glm53`.
- Core architecture tests passed `3/3`: `test-batch-alloc`, `test-llama-archs`, and `test-glm5next-sparse`.
- CUDA operation regressions passed: `KPOOL_EXPAND` `2/2`, indexed FlashAttention `11/11`, and fused MoE down reduction `24/24`.
- Sequential GLM5NEXT MTP comparison passed on CPU and all six CUDA devices. The MTP top-k reuse A/B produced the same greedy decisions, the reuse graph omitted indexer K/G and pool scoring after its first iteration, and its RTX 4090 compute-sanitizer run reported zero errors.
- The backend-resident MTP loop matched host-selected tokens, probabilities, final text, and acceptance. It passed a simultaneous two-slot server run and improved the measured generation rate by 1.8% at short context and 1.27% after a 55K-token prompt in the controlled fallback A/B runs described above.
- End-to-end pool-cache validation against the local GLM-5.3-Flash model produced identical cached and uncached logits (`max abs = 0`), identical argmax output, and a successful multi-stream restore result.
- Documentation-only commits are intentionally excluded from the numbered implementation list.
