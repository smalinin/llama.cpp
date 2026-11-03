# DS41 indexer: move pre-pooling work to the score producer

Status: implemented and validated as explicit opt-in/defaultOFF, awaiting result review. Based on the [fresh diagnostic](../experiments/1017-hc-long-indexer-20261103/REPORT.md), my_maya9b74cc6e6 plus the already validated HC-sharing delta. No new defaults in this design.

## Observed problem

At the tested layer split, layer20 LIGHTNING_INDEXER and its final TopK run on CUDA2. The candidate branch copies the complete contiguous F32 score to CUDA3, runs an explicit CONT there, then POOL_2D max over8 rows, pin/add/block-TopK/scatter/repeat. Its dense mask is consumed locally by new queries24/28 on CUDA3 and copied to CUDA4 for32/36. Reused final TopK for intervening layers does not replace this block selector.

A full98K diagnostic sees72 candidate frames,8.782GB logical full-score payload and the same dense-mask payload. The pooled scores are8x smaller. This byte ratio is not a predicted8x end-to-end speedup. No matched Maya speed claim follows from this audit.

## First prototype: placement only

Reuse graph_get_cb and scheduler set_tensor_backend/supports_op, as existing layer-local HC/index hints do. Keep every operation, dtype, source range and order. Start explicit opt-in/defaultOFF, restricted to deepseek41, supported CUDA layer split and the actual source20 fused-indexer geometry established in the report. Unsupported mode/backend/callback follows the existing path. Do not create another CUDA stream/kernel/buffer manager or change scheduler iface/events/copy-slot policy.

Pin the complete pre-pooling chain to the verified score-producer device:

1. Existing CONT(index_score), even though input is contiguous in this fixture.
2. Existing padding FILL/CONCAT when n_pos is not a block multiple.
3. Existing POOL_2D max, via its candidate_block_score callback.

Pinning only POOL_2D is insufficient: CONT could still run on CUDA3 and retain the full-size outbound transfer. Add explicit callback names for the existing pre-pooling nodes; do not delete the CONT as part of this experiment. Confirm the actual score producer agrees with the selected owner and that every pinned op is supported. Do not infer actual producer solely from an arbitrary layer number on another topology.

Keep pin/add/block-TopK/scatter/dense expansion on their existing destination in the first experiment. Expected edge: pooled candidate_block_score CUDA2->CUDA3, replacing the full score copy. The later dense candidate_mask CUDA3->CUDA4 remains a separate issue. If allocation/placement shifts additional nodes, measure and document those shifts rather than assuming the intended edge is active.

## Semantics and ownership

- Preserve block8, selected2048 blocks, final per-query TopK2048 positions, causal masks, newest-visible-block +inf pin, padding -inf, existing CUDA max/all-inf/NaN/tie behavior and output order.
- No conversion of block TopK to source-layer final TopK. Do not truncate the allowed candidate set to2048 rows.
- Preserve tensor roots, widths/ne/nb/offsets, n_copies/cur_copy, current producer and consumer events, allocator lifetime and capture/replay validity. Never deduplicate by name/address alone.
- Existing small/partial/verify/decode modes must preserve fallback or pass their own exact gate. Multi-stream/media/adapters/other quant/KV/model support is not inferred.

## Acceptance

1. Before/after graph placement, data/ranges and actual copy API trace: prove the large score copy disappears and the smaller block-score copy reaches the same selector; check no new large reverse edge. Record full-size source/destination pointer identity in a bounded external probe if needed; the present diagnostic only constrains shape/backend/consumer interval.
2. Operator controls: candidate disabled/enabled boundary, partial block, all-invisible rows, newest visible block, ties, nonfinite inputs and non-contiguous/fallback cases. Preserve existing max semantics, not an idealized alternative reference.
3. Existing target/draft logits/taps/seq/global/partial-state gates, full-prefix restart/reuse, zero/partial/full acceptance and rollback, controlled failure/full retry. NLL text/code: transport-only expected exact; +0.01nat/token diagnostic threshold is not permission to accept unexplained transport differences.
4. Fresh no_alloc and actual176128/172032+16 with per-GPU512MiB sampled floor; lower ctx if admission fails. No extra concurrent workload.
5. Clean OFF/ON/OFF PP512/8K/98K with fixed raw IDs, B8192/U512/P1/TopK2048, Q3/MXFP8 Engram/Q8KV/BF16K3/full/p_min0.5 and exact placement.256 outputs for TG, not16-output profiling samples. Separate NSYS diagnosis from native timings and compare complete PP costs.

## Second candidate, only after first review

If dense-mask transport is still material, carry the existing 0/-inf block keep representation and expand it on each actual consumer device. This requires a separate per-device ownership/lifetime design and tests for masks, padding, copy slots and graph reuse. It does not require replacing CUDA numerical kernels or importing another engine wholesale. CPU pin generation/upload is another measured candidate, not folded into the first prototype.


## Implementation and measured review result

[Prototype report](../experiments/1017-candidate-pool-local-20261103/REPORT.md). User requested commit and continuation; previous work committedc3c91dda8 before implementation. The first post-pool ADD callback sees the complete chain and performs an all-supported check before any assignment. Fused score owner is validated against the actual persistent index-key buffer and model layer device; source20 and next query24 remain deliberately bounded. The source score itself is pinned to that verified device. The first ADD stays at query24 owner to retain the measured selector placement. A final bound check rejects exhausted next-source search before dev_layer access.

All planned data/state/memory/native gates completed for the frozen fixture, with measured/guard-final identities kept separate. The mutable OFF/ON fixture has one documented retained-capacity warning; fresh constantON passes clean. Initial operator implicit-allocation routing was rejected and corrected by actual device assertions. PP98K+1.46%, no TG gain established (98K-1.07%singleton), combined no_alloc reserve-1073MiB@176K,actual minimum761.25MiB. Default remainsOFF. The second block-keep candidate is not implemented.
