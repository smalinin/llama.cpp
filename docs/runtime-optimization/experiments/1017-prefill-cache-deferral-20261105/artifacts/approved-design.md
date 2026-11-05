# DeepSeek41 prefill: deferred remote KV replica writes

Status: proposed for review after read-only ownership audit; implementation not started. Initial prototype OFF. This is a model-graph ordering change using the existing deferred-cache-update map, not a scheduler/event/allocator rewrite. No measured speedup is claimed.

## Evidence and problem

Runtime source is the unchanged f796e544f implementation, documentation HEAD cf1e5a461, new-host Threadripper PRO3945WX/current UUID order. [Ownership report](../experiments/1017-kv-owner-audit-20261105/REPORT.md), full raw metadata and source snapshots are retained there. Queue4x, private copies8 and shared gate/up6/29 co-location already failed their native PP screens.

The selected 512-token chunks show eager source20 replica work before the source GPU continues:

| Cache root | Actual destination | Cache shape/type | First full-tail reader at prefix4096 |
| --- | --- | --- | --- |
| cache_k_l20 | CUDA2 | [512,100352,1,1], Q8_0 | graph30, k_all-20 CONCAT |
| cache_k_l23 | CUDA3 | same | graph35, k_all-23 CONCAT |
| cache_k_l30 | CUDA4 | same | graph40, k_all-30 CONCAT |
| cache_k_l38 | CUDA5 | same | graph49, k_all-38 CONCAT |

Writes occupy graphs26/27/28/29, then execution returns to CUDA2 at graph30. The preceding LID replica writes run onCUDA3/4. ROPE and Hadamard of comp_kv-20 then execute onCUDA4, despite the source layer being onCUDA2. The transformed F32 latent is copied from its actual CUDA4 owner to CUDA2/3/5. CUDA4 owns real cache writes; these are not no-op view splits.

Those physical GPU numbers describe this captured configuration, not constants for implementation. Root/owner mappings are derived from public buffer metadata and exact root identity inside each ubatch. The same cache allocations persist across captured chunks; protected input-copy slots rotate and repeat. Numeric row-index values were not read, so the audit does not prove writes disjoint merely from prefix offsets.

Existing LLAMA_DSV41_DEFER_CACHE_UPDATES is already default ON for decode/verification, but requires is_dsv41_decode. It queues remote writes until the first reader layer via cache_updates. Blindly removing that gate is invalid: KV-only prefill returns at layer20 with cache_updates.empty(), while future remote replicas must still include those prefix rows.

Maya's chunk workers publish rows into bounded stage slots and each reader pulls them on its stream, acknowledging completion before stage reuse. This suggests delaying work to its reader, but this proposal keeps our existing SET_ROWS/quantization/transport semantics instead of importing Maya's kernels, formats or flag protocol.

## Proposed bounded prototype

1. Add explicit opt-in LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1; unset/0 retains current prefill. Keep the existing decode flag and conditions unchanged. No default promotion/deployment is part of this design.
2. Initially restrict to DeepSeek41 text token input, causal flash attention, one sequence/slot, layer split, fully GPU-offloaded attention/model, and short bounded prefill ubatches up to512. Mixed sequences, embeddings/media, partial offload, callbacks, unsupported cache storage and unproven shapes use the original eager graph. Q8_0 is the first full-model validation cohort; F16 needs its own admission/quality evidence before the prefill opt-in is allowed for it.
3. Reuse write_cache and cache_updates. Keep local writes eager. Queue only existing remote CSA/LID SET_ROWS operations; keep source values, indices, destination, order for any common destination, quantization and all existing copy/event/slot handling unchanged. Determine reader/source devices from model/cache ownership, never hardcode layers20/23/30/38 or GPU IDs. Unrecognized ownership stays eager.
4. For full prefill, expand queued operations immediately before their first actual local reader, using the existing map/loop insertion point. Preserve the existing inpL/pre_mix expansion at device boundaries so older ancestor work is not dragged onto a later device. No duplicate/drop of writes, no new alternate source tensors.
5. For KV-only early return, after the required source-layer state/KV/index work is built, expand every remaining pending replica update, then clear the map before the existing empty-map assertion and return. Flush only after source values/indices exist and before the next ubatch/context transition. Preserve n_outputs0 and the empty-map assertions. No delayed work may survive the graph lifetime.
6. For every ordinary exit, assert pending updates empty. Failures/reset/rollback retain the scheduler's existing synchronization and state-restoration contracts. Do not introduce cross-graph pending jobs or asynchronous host tasks.

## Ownership and ordering contract

- Each remote write has the same original values/indices/cache root. SET_ROWS remains an in-place write into its persistent cache allocation.
- The graph retains explicit producer references until the last delayed use. Let the existing allocator extend live ranges and account for copies; never replace tensor data pointers or infer aliasing from names/address reuse.
- Existing source-stream peer-copy ordering and destination event waits stay intact. Input-copy slots must retire before reuse. No scheduler fence or wait_input_copy removal.
- No consumer may read its cache replica before its queued writes. In KV-only chunks there may be no local consumer, so explicit flush is mandatory.
- No graph input, output tap, mutable compressor state or source operand is mutated to make deferral work. Ordinary state snapshots, recurrent rollback and DSpark tail seeding must see the same bytes as the eager path.
- Replica SET_ROWS are idempotent only under the existing row-index plan; do not use assumed disjoint prefix ranges to justify overlap. Repeated prefix/reset/failure cases must be tested.
- If placement remains poor, producer-local transform pinning is a separate candidate. Do not combine it silently with the first deferral experiment or promise it will follow from graph heuristics.

## Validation before adoption

1. No-alloc graph audit on the frozen layout: every old CSA/LID write remains exactly once with the same value lineage/indices/destination/type; local writes stay eager, remote writes precede the first reader, KV-only flush empties the map. Capture actual buffer owners and source transform execution after scheduling.
2. Compute/workspace admission for ctx100352/B8192/U512 and512MiB free on every GPU, based on measured per-device overhead, then monitored real loading/inference. If required, reduce context identically in both arms with fresh baselines; metadata estimates do not replace actual memory evidence.
3. Raw OFF/ON target logits, exported taps, compressor/ring/KV snapshots, NLL on fixed text/code prefixes, restore and rejected-window rollback must match exactly. This changes scheduling only; the generic NLL tolerance is not a license for drift. Cover KV-only/full-tail transitions, source-ratio boundaries,511/512/tail508/short decode, copy-slot wrap and DSpark processing after prefill.
4. Existing scheduler/lifetime fixtures plus full-model failure/abort/retry/reset, repeated uncached prompts and continuation after restore. Every pending write must execute after retry; no buffer mismatch/OOM/stale state or forced kill.
5. Clean native OFF/ON/OFF, PP512/8K/98K, sustained TG256 with the same DSpark/max5/pmin0.5/full/KV/placement and frozen token IDs. Trace activation separately; no evaluation callbacks/profiler in the speed baseline. If gain is small or negative, preserve the report and leave prototype OFF or revert its code.
6. Other models, media, parallel>1, CPU expert cache/staging and broad joint auto-fit are not granted by this validation. No merge/default change without review and explicit user request.

## Approval boundary

This introduces a new prefill graph ordering and a KV-only flush path. The repository [AGENTS.md](../../../AGENTS.md) requires: "If the change is large or introduces a new pattern, PAUSE and ask the user for confirmation". The existing CPU lookup approval covered a different mechanism. This review covers only the bounded prototype and its checks above; it does not authorize scheduler guard removal or default promotion.
