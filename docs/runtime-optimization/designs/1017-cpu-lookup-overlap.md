# CPU lookup overlap within the existing pipeline scheduler

Status: approved by user 2026-11-05, implemented and verified; source base aa33d0490, production libraries from the new-host baseline kept intact. The user requested fixing the measured PP/TG gaps and continuing with prefill first. This document makes the CPU/GPU ordering change reviewable before implementation.

## Evidence and objective

The new-host telemetry maps the largest sampled prefill cudaEventSynchronize caller to libggml-base offset0x3fd2a. Disassembly identifies the no-input split transition fence in compute_splits, before split input copies. CPU graphs on this model contain only GET_ROWS plus reshape: token embedding, Engram1 and Engram14. These source weights and input row IDs are CPU-resident. The current fence makes the host wait for the previous GPU before starting an otherwise independent CPU lookup. The allocation/input-slot wait samples are much smaller. Sampled durations are not native time shares; a counters-only native run will measure this branch separately.

The prototype may omit that one previous-GPU host fence only for a proven isolated CPU lookup split. No global synchronize removal, new thread, stream, allocator policy, CUDA arithmetic, model metadata or public API.

## Difference from the earlier Engram-owned lookup screen

[E11](../evidence/reports/E11.txt) retained the zero-input transitions and found no meaningful PP gain from a separate CPU producer/pool. This candidate does not repeat that worker/prologue change. It targets the preceding conservative GPU-to-CPU fence and reuses the existing protected host-stage ring, with no new buffers or threads. E11 does not prove that this different ordering is safe or faster.

## Eligibility and ownership proof

All conditions are required, otherwise use the original fence:

- Explicit startup opt-in `GGML_SCHED_CPU_LOOKUP_EARLY=1`, default OFF; async inputs prepared, n_copies>1, no eval callback.
- Split has zero cross-backend inputs and executes on the CPU backend. Its compute operations are non-inplace GET_ROWS from immutable CPU host weight storage using protected CPU input row IDs. Other nodes are read-only views of those results. No state mutation, adapters, exported CPU output, CPU expert matmul or GPU-sourced index tensor.
- All non-CPU compute backends/operations are discrete CUDA with device-backed tensors. Reject direct GPU access to host compute storage, integrated/unified memory paths and staged MoE operations. CPU results must reach GPU only through the existing per-copy host-stage and destination copies.
- All backend buffer types must be distinct: gallocr otherwise shares scratch across their streams. The existing set_inputs_async API already enforces this; reuse its rejection rather than duplicating the guard. Shared-type/virtual layouts retain the original transition fence.
- Each CPU lookup result exported to a GPU has the existing INPUT/OUTPUT-protected host-stage allocation for that copy slot, separate from CPU scratch. Its H2D source remains alive until the slot-retirement events complete. CPU scratch may be reused only after the synchronous CPU-to-stage copy has finished.
- Input preparation/slot retirement, destination wait, GPU producer-to-consumer dependencies, CUDA source-stream copy ordering and allocator fallback synchronization remain unchanged. A CPU lookup cannot read GPU-produced data or overwrite host memory that an in-flight GPU/DMA still reads.
- Keep a conservative previous-graph isolation gate, or equivalent proof, across rebuild/reuse transitions. An unproven predecessor, layout growth, failed compute or callback path falls back to the full original ordering. No inference from tensor names or equal data addresses.

The implementation must stop and revise this design if it needs new persistent scratch, a worker thread, cross-split data relocation or changes to the existing copy-slot/event contract. It must not simply skip every CPU fence because n_inputs==0.

## Validation

1. Counters-only current binary run, no CUPTI/backtrace: identify wait callsite and following CPU GET_ROWS split, slot retirement vs compute transition, CPU lookup time, PP8K/98K and continuation. Retain native controls, source/library hashes and512MiB VRAM floor on the new host.
2. Existing external scheduler harness or bounded reproduction outside tests/: CPU GET_ROWS between two CUDA splits, more than four consecutive chunks, retained staging/destination buffers. Positive exact output and per-slot lifetime; negatives GPU-derived row IDs, mutable weights, direct host reads, CPU state/FFN, callbacks, single-copy and staged MoE.
3. Shape changes, allocator growth/fallback, reset/rebuild/reuse, failure/retry, CUDA capture/replay and cancellation preserve original ordering outside eligibility.
4. Full target/draft logits/taps/KV/state and rollback exact toOFF. This is scheduling only, no arithmetic relaxation or NLL allowance. Native OFF/ON/OFF PP512/8K/98K and TG, unchanged placement/acceptance, actual memory and clean exit. No automatic promotion or new implementation commit.

A failed or negligible speed experiment is retained in the knowledge base and the runtime prototype is reverted. TG six-query matmul/MoE remains a separate next candidate, not mixed into this patch.

## Default promotion requested 2026-11-05

After exact native/precision gates and review, the user requested default ON and a commit. Only the constructor default changes: unset/1 enables the existing guarded path, 0 disables. All isolation, staging, callback, predecessor and async-input guards remain unchanged. The original opt-in validation above is historical; promotion fixture/build results are recorded separately in the [report](../experiments/1017-cpu-lookup-overlap-20261105/REPORT.md#default-on-promotion).
