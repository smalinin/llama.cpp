# HC copy sharing: bounded prototype design

Status: approved by user2026-11-03; bounded prototype validated; default ON promoted at explicit user request. Base: my_maya9b74cc6e6; runtime6d038260e. Evidence: [actual alias audit](../experiments/1017-m2-large-context-20261103/REPORT.md).

## Goal and scope

Keep one destination buffer/copy for an HC root and its full contiguous reshape when both are inputs of the same scheduler split. Preserve all arithmetic, types, strides, masks, KV/draft state, placement and transfer order. The measured prototype was opt-in, `GGML_SCHED_COPY_RESHAPE=1`, unset/OFF initially. After full gates the user explicitly requested default ON; unset/1 now enable the guarded path,0 disables it. No CLI/model-file change.

The recorded8K audit shows32 pairs/1.342GB repeated API bytes at l_last-6 CUDA0->1 and l_last-14 CUDA1->2. Physical DMA and cost attribution are a separate read-only measurement. Those byte counts do not predict PP improvement.

## Eligibility

All conditions must hold; otherwise retain the original scheduler branch:

- Pipeline scheduler with n_copies>1, protected copy slots and supported CUDA->CUDA transfer. Same source backend for root/view; same destination backend and current copy slot.
- Direct GGML_OP_RESHAPE of a non-view GGML_OP_DSV4_HC_POST root, both F32 and contiguous. Equal element count and full byte count, zero view offset; no transpose/permutation/subrange or nested view. No user graph-input or exported-output tensor.
- The root already appears explicitly in the current split's input list, with its existing canonical destination copy. That copy has the existing INPUT/OUTPUT protection. Do not synthesize an earlier root copy, change its enqueue order, or reuse a root found only in an earlier split.
- No graph nodes need different storage for mutation; retained canonical OUTPUT protection must block allocator in-place reuse through the new view. The initial prototype does not enable unprotected n_copies1 mode.

Eligibility uses tensor identity, operation/layout and split ownership before allocation. It never uses names, equal runtime addresses or recorded layer/GPU constants. Identical pointers at different graph times are not an alias proof.

## Allocation and execution

1. Leave canonical root handling unchanged: existing hv_tensor_copies[root,destination,cur_copy], split input and source dependency remain. The current allocator/input-bindings code owns its storage; no manually assigned data pointer or new persistent GPU buffer.
2. At a matching reshape consumer, construct a destination reshape view of the canonical destination root using the requested ne/nb. Reuse a local same-split view if needed. Do not store this view as a global hv_tensor_copies entry for the source reshape: a later split must take the original fallback and cannot silently inherit broader reuse.
3. Replace only that consumer's source with the destination view. Do not append the reshape as a second transfer input. Nonmatching/earlier consumers stay unchanged, including any independently required copy in a different split.
4. Insert the destination view as an allocation-only node in graph_copy before its first consumer, after canonical input allocation. Count it in graph capacity/backend IDs. A bare src/view pointer is insufficient: gallocr counts n_views from graph nodes and releases views through their consumer count. Do not skip normal ggml_backend_view_init or mutate an already captured graph after allocation.
5. Backend graph optimization/fusion sees the final source/view relation before optimization. Existing alloc_deps, graph properties, address/layout invalidation and CUDA graph capture/reuse remain in force. No new CUDA kernel, stream or event is needed for a reshape.
6. Do not remove fences. The existing canonical root transfer completes under its source copy-event/destination stream wait. Both consumers read that buffer on the same split backend. Existing per-backend/per-cur_copy retirement protects overwrite by the next pipeline chunk.
7. Temporary CPU view bookkeeping lives only for this graph split/build. Reset/rebuild/cancel/failure cannot retain old alias entries. Existing graph rebuild selects the matching cur_copy; no view can point at another slot's allocation.

The intended runtime change is limited to split-graph construction/allocation bookkeeping in ggml-backend.cpp. Do not change compute_splits, CUDA kernels, allocator reuse policy or public API. If implementation requires wider ownership/event/cache changes, stop this prototype and update this design first.

## Validation order

1. Source/graph eligibility check on the real frozen boundary, including same-split membership. Read-only observer can verify copy queues and backend split launches before runtime changes.
2. Existing external scheduler harness, outside tests/: protected CUDA0->CUDA1 same-split root+reshape exact values, one copied range, shared destination storage. Negative controls: flagOFF, nonzero offset/subrange, noncontiguous, nested view, different root/type/backend/split, user input/output, root missing/late, n_copies1. Verify fallback actually runs.
3. More than four successive pipeline chunks, slot rotation, shapes1/4/6/7/508/511/512/513, rebuild/reuse and controlled compute failure/retry. No stale allocation/view; guards remain enforced. CUDA graph eager/capture/replay must see consistent alias properties.
4. Reuse the existing target/draft M2 state/logits/rollback harness. Exact comparison of target/draft logits/states/taps, K3/K5/confidence, append/restore/replay and rejection/partial/full acceptance. This is a transport-only optimization; arithmetic changes are out of scope.
5. Fresh default/explicitOFF/ON no_alloc with bound draft and unchanged actual tensor placement. Actual full load respects per-GPU512MiB sampled floor. Reduced copy memory is measured; no_alloc alone is not a capacity pass.
6. Native OFF/ON/OFF full uncached PP512/8K/98K and TG256, frozen Q3/MXFP8 Engram/Q8KV/BF16K3/full/p_min0.5/B8192/U512/P1/TopK2048. Keep the same context/layout and confirm eligibility/copy reduction in a separate diagnostic. No profiler/callback/interposer in timed runs. Check actual memory, exact IDs/counters, exit0 and warning-free cleanup.
7. Keep or reject by complete measured cost and correctness. No automatic commit of new implementation/default change. Q2/Q4, unprotected single-copy, cross-split aliases, CPU paths, arbitrary views and other backends remain excluded.

## Review questions and required evidence

The approved prototype additionally limits the shared view to a non-inplace RMS_NORM consumer and rejects HC roots with unknown/non-read-only readers. This narrows the approved scope. The allocation-only view is tracked in graph_copy/node backend IDs without adding a new global tensor-copy key. CPU/GPU numerical gates and capture/replay passed; clean native OFF/ON/OFF showed PP+3.8..4.3% and640MiB reserve savings. The user requested default ON. See the [experiment and promotion](../experiments/1017-hc-copy-sharing-20261103/REPORT.md).

- The same-split bound deliberately gives up some possible sharing to preserve existing dependencies. If the recorded real copies are in different split instances, this prototype is not useful; do not widen the bound to chase a gain without a new design.
- Adding views only to allocation graph is a scheduler/allocator pattern change. Before implementation, review the root lifetime, n_views bookkeeping, per-copy ownership and CUDA fusion/capture behavior above.
- After correctness, measure physical copy correlation and matched native speed. An operator/API byte reduction can still produce no useful end-to-end improvement.
