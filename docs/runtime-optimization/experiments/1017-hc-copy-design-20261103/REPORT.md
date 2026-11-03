# HC copy sharing: design and historical cost screen, 2026-11-03

Status: read-only analysis, prototype not implemented. Runtime my_maya9b74cc6e6, implementation6d038260e. The previous M2 residual/large-context results are committed in9b74cc6e6; notebook plans inbc18c40. No default promotion or new GPU process in this step.

## Proposed bounded change

[Design](../../designs/1017-hc-copy-sharing.md): opt-in GGML_SCHED_COPY_RESHAPE=1, initially OFF. One canonical destination copy for an HC root and direct full contiguous zero-offset F32 reshape in the same scheduler split, protected pipeline copy slots only. Preserve existing source dependency, copy events, destination wait, cur_copy ownership and input-buffer overwrite guard. Insert the destination view in allocation graph before its consumer so gallocr counts n_views normally. Do not globally alias copies across split instances, use raw pointer equality as identity, assign buffer pointers manually, or enable unprotected single-copy mode.

The design is saved for review before implementation. AGENTS.md:82 requires confirmation for a new pattern. The additional allocator view node and shared destination lifetime make this such a change. No production implementation is being held behind a new ad hoc policy; this is the repository instruction applied to the proposed scheduler pattern.

## Historical Nsight reanalysis

CPU-only, read-only analysis of the immutable captured8K request: /tmp/dsv41-m2-residual-profile-20261103/sidecar/capture.2.sqlite. This is the previous M2 ctx100352 capture, not the later ctx176128 pointer audit and not a new performance run. SQLite SHA256, exact query script and selected API/CUPTI records are saved in artifacts.

Selection: cudaMemcpyPeerAsync runtime API inside ubatch.target_prefill; correlated memcpy record with full HC size41943040bytes or final41615360bytes; immediately following CUDA1/CUDA2 backend graph launch on the same host thread. Group by that next launch. All32 selected groups contain exactly two API calls. This confirms that the observed full-size pairs fit the proposed same-split/next-launch bound. Correlation to tensor names is supported by matching sizes, backend sequence and the independent pointer audit; this historical SQLite lacks the later observer's source/destination pointers, so it does not independently prove pointer identity.

| Next backend | Groups | Peer-copy API calls | CUPTI memcpy records | Record byte sum | API host duration sum, ms | Copy record duration sum, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA1 | 16 | 32 | 64 | 2683043840 | 5.042 | 12.628 |
| CUDA2 | 16 | 32 | 64 | 2683043840 | 5.290 | 7.587 |

Each selected API correlates to one D2H record on source device and one H2D record on destination. Record byte sums therefore count both legs; they are twice the logical API payload. Combined record sum5366087680bytes is not the potential saving1341521920 logical bytes established by the independent alias audit. Do not add them or infer measured PCIe bandwidth from these records.

The runtime reports the two API calls before one destination backend graph launch. Existing CUDA source copy-event/destination wait and protected root copy storage remain the dependency contract; neither API timing nor copied bytes justify removing a fence. The trace records transfer activity, but its summed durations do not equal complete staging/synchronization latency or an exclusive wall-time saving. Native OFF/ON/OFF is still required, and gain may be small despite substantial duplicate payload/allocation.

## Next work after review

1. Confirm this bounded design, then implement only split/allocation bookkeeping under default-OFF selector.
2. Verify structural eligibility and exact transport on the existing external scheduler harness, including negative controls, more than four chunks, source lifetime, cur_copy rotation, rebuild/capture and failure/retry. No new tests/* files.
3. Reuse target/draft state/logits/rollback harness and fresh no_alloc/actual memory gates, floor512MiB on each GPU.
4. Clean native same-context/layout PP512/8K/98K/TG256 OFF/ON/OFF. Diagnostic alias/copy count and CUDA correlation run separately. Keep or reject based on full cost; no default promotion or commit without user request.

The prior large-context gate and numeric M2 gates remain valid baseline evidence. This design/cost screen does not replace candidate correctness, actual memory, native speed, matched Maya or Q2/Q4 coverage.
