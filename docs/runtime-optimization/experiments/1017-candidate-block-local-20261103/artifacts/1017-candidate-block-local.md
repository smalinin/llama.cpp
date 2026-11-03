# DS41 candidate mask: compact block transport and local expansion

Status: proposed for design review, not implemented or benchmarked. Base: my_maya b3377774b. Previous [producer-local pooling experiment](../experiments/1017-candidate-pool-local-20261103/REPORT.md) remains opt-in/default OFF.

## Problem and bounded hypothesis

The measured 172K request had 134 wide candidate-mask CUDA3->CUDA4 copy API requests, 26.700292096 GB of logical payload. These bytes are API payload, not physical DMA or critical-path time. Pool placement removed the wide score transfer but did not remove this dense-mask edge. The current graph already has an F32 `candidate_keep` containing one 0/-inf value per block before REPEAT expands each block to eight positions.

Transfer the existing block representation and expand it on each consumer GPU using the same ggml operations. The aligned representation is eight times smaller; no eightfold engine speedup is predicted. The new flag is `LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1`, initially explicit opt-in/default OFF. Keep the previous pool flag independent and unchanged. No model-file, CLI, sampling, CPU spin policy or user preset changes.

## Numerical contract

Keep the source score, padding with -inf, max pooling, newest-visible-block +inf pin, block TopK and SET_ROWS unchanged. Candidate selection is 2048 blocks of eight positions, followed by independent per-query TopK2048; carrying source final TopK would change the algorithm and is forbidden here.

With N visible/history slots, Q queries, B=ceil(N/8), S=1:

- SET_ROWS root: F32 [1,B,Q,S]. Existing view `candidate_keep`: [B,Q,1,S], contiguous in this bounded case.
- Transport a complete block tensor/view with its exact ne/nb and zero offset. Do not transfer a strided dense subrange or infer compatibility from equal bytes.
- On a validated consumer device: reshape to [1,B,Q,S], REPEAT to [8,B,Q,S], reshape to [8*B,Q,1,S], and apply the same final zero-offset trim to N. Preserve padded row stride when N is not divisible by eight.
- Add this F32 dense mask at the existing score ADD. Do not move masking before scoring or change fused LID arithmetic. The current nt=1 pre-score masking path is excluded and stays unchanged.

Only 0/-inf block values move. NaN/inf/tie handling in score pooling and TopK must remain that of the existing CUDA kernels. Re-expansion is a byte-preserving reshape/repeat, not a new reduction or quantization.

## Eligibility and graph construction

Start with the previous bounded fixture: deepseek41/source20/ratio1/block8/candidate2048/final2048, CUDA layer split, causal fused LID and offloaded KV, one sequence/stream, token input, 6<Q<=512, no eval/abort callback or LoRA. Only enter if candidate_pin exists and all consumers use the same candidate domain/shape. Tiny verification/decode, no-candidate contexts, media, CPU/other backends, parallel>1 and unsupported topology keep the original graph.

Use actual persistent buffer devices and supported operations, not recorded CUDA ordinals. Validate source index-key owner and the query projection/score owners for every consuming index-source layer. Shared source keys can be on a different device from queries; do not mistake the key owner for every consumer's owner. In the frozen fixture consumers24/28 are on CUDA3 and32/36 on CUDA4. These ordinals are observations, not a routing table.

Freeze the selected mode for a context before reserve/build; propagate graph-shaping eligibility through graph parameters and their reuse comparison. Runtime and no_alloc/fit must agree. Do not switch environment in a loaded context. Check complete eligibility/support before assigning backends; no half-built optimized graph after a late failed check.

Retain `candidate_mask_carry` as the legacy fallback. In the eligible graph retain the block tensor separately, and build the existing expansion lazily for each consuming device. A graph-local map may share an expansion only for the identical block tensor identity, device and complete N/Q/S/layout domain. The map is destroyed with the graph. It is not a runtime cache of GPU pointers or a cross-context tensor registry. Explicitly assign the REPEAT and score ADD to the verified consumer backend; leave selector/pin/scatter at the existing selector owner. Verify the resulting graph rather than relying on backward placement propagation.

Do not build/expand the unused legacy dense branch into the eligible executable graph. Otherwise a later consumer could still copy the full mask and invalidate the hypothesis. Keep the old path intact when ineligible. Splitting the helper into block selection and row expansion is acceptable; it must not change source TopK or score dispatch.

## Ownership and synchronization

All payload allocations remain scheduler/gallocr-owned tensors. No manual data pointers, new CUDA streams/events, persistent GPU buffers or scheduler/allocator changes are included.

The block source is an explicit dependency of every local REPEAT. Existing scheduler source dependencies keep it live until transfers finish. Existing destination copies and per-cur_copy INPUT/OUTPUT protection own transported blocks. Each dense expansion is a normal graph node on its consumer device; its consumers keep it live. A retained graph-local tensor reference does not by itself prove allocator lifetime: inspect the split graph and actual protected copies.

Do not alias dense masks on different GPUs or across copy slots. Do not remove producer completion, destination wait, copy-slot reuse or input overwrite fences. If existing scheduler rules cannot express the required lifetime safely, stop and revise this design before widening the implementation.

Graph rebuild, capture/replay and parameter reuse must see all expansion nodes and dependencies before allocation/optimization. Cancellation, reset, failure/retry and context destruction must not retain a map from the previous graph. More than four queued chunks must exercise slot rotation; one successful synchronous launch is insufficient.

## Validation and decision sequence

1. External operator harness outside tests/: legacy single dense expansion vs block transport/local expansion on two actual verified CUDA devices. Check whole mask bytes and masked scores/TopK, padded strides, B<=2048 no-selection fallback, boundary N=16384/16385/16391/16392, and actual long sizes. Query widths1/6/7/511/512/513 exercise eligible/fallback boundaries. Cover finite/ties/all-invisible/nonfinite source scores, newest visible block, multiple source patterns and at least32 changing inputs on a fixed captured graph.
2. Negative eligibility controls and more than four pipeline chunks: no stale blocks/masks across slots, rebuild, graph reuse and failure/retry. Explicit backend assertions after allocation; implicit allocation may discard manual hints, as the previous harness demonstrated.
3. Fresh metadata OFF/ON at100352/176128, both with fixed previous pool flag, actual model/KV placement and buffers unchanged. Record compute reserve per GPU. New local dense allocations may increase consumer VRAM; an eightfold transport reduction is not an eightfold memory reduction. Preserve sampled free floor512MiB on every GPU; reduce context if required before real loading.
4. Existing target/draft raw logits, taps37/38/39, seq/global/partial states, K3/K5/confidence, append1/7/511/513, prefix restore/reuse, zero/partial/full acceptance, rollback and controlled seed failure/full retry. Two text/code corpora, NLL diagnostic +0.01nat/token retained; this transport-only change is expected to be exact and unexplained deltas block promotion.
5. Separate actual copy proof with full pointers/type/ne/nb/range/consumer/slot/device: wide dense-mask cross-device copies disappear and compact block transfers reach the intended local REPEAT. Record counts/bytes, padding, tiny fallbacks and any new reverse edge. Logical requests and CUPTI/physical traffic remain distinct.
6. Clean native OFF/ON/OFF full uncached PP512/8K/98K and TG256 with fixed IDs, Q3/MXFP8 Engram/Q8KV/BF16 DSparkK3/full/p_min0.5/B8192/U512/P1/TopK2048, identical placement/context and no callbacks/profiler/calibration. Hold CANDIDATE_POOL_LOCAL=1 fixed to isolate the new mechanism, then short controls with pool0 to establish flag independence. No claim of two independent speedups from a combined result.
7. Actual176128 with172032+16 and repeat8K if admission passes. Archive exact outputs/counters, clean shutdown and sampled memory. Default remains OFF until user review. Matched Maya and Q2/Q4 remain later tasks, not implied by these gates.

## Scope boundary for review

This step changes graph representation and per-device expansion ownership. It does not redesign the scheduler, alias transport buffers, move block selection to CPU, remove pipeline fences, extend verification eligibility or enable the prior pool experiment by default. If that boundary cannot be maintained, return to design review before implementation.
