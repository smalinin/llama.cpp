# KV owner and slot-lifetime audit, new host 2026-11-05

Status: read-only audit complete after residual commit `cf1e5a461`; external plans saved separately as`b970cfc`. New proposal is ready for review, not implemented. Runtime source, installed libraries, defaults, model files and user presets remain unchanged. No second commit/deploy.

## Result

The remaining transition before `cache_k_l30 (view)` protects a real in-place Q8_0 KV write. It cannot be discarded as metadata-only. This audit also found that the source20 compressed latent's ROPE and Hadamard run onCUDA4 after eager LID replica writes, even though the logical source layer is onCUDA2. The resulting F32 payload is then copied from its actual CUDA4 buffer toCUDA2/3/5 for CSA replica SET_ROWS.

The observed selected-chunk submission chain is CUDA2 source/index work -> CUDA3 LID -> CUDA4 LID plus ROPE/Hadamard -> CUDA2 CSA -> CUDA3 CSA -> CUDA4 CSA -> CUDA5 CSA -> CUDA2 source-state/indexer/attention continuation. This is an observed graph/owner order, not a measured critical-path time decomposition. GPU IDs are observations for the frozen layout, not implementation constants.

## Frozen method and checks

Same new-host Q3/MXFP8 Engram/Q8_0 K/V/TopK2048/ctx100352/B8192/U512/P1/threads12, all15360 target expert bundles on GPU, existing BF16-named mixed-format DSparkK5/full/pmin0.5, default launch queue/copies4/default CPU lookup ON. No cache/staging/mmproj. Source implementation remains f796e544f; cf1e5a461 changes documentation only. [Expanded args/env and identities](artifacts/counts-results.json), [observer build](artifacts/build.json).

The external observer forwards original operations, copies, status and existing events. It adds no GPU events, CUDA API interposition, eval callbacks, reads of numeric tensors, or math. It records node/source/view-root pointers, shapes/strides, flags, op parameters, buffer type/device/base/size/usage and execution backend. Public root-buffer device metadata avoids `sched_get_tensor_backend` on arbitrary operands: that getter uses find_or_insert and could mutate scheduler metadata.

Selected target ubatches cover positions0/512/1536/2048 with KV-only layer20, and full-tail4096/4608/6144/7168/7676, including508-token tail and copy-slot reuse. There are374 complete split snapshots and36 traced source20 cache writes. Child source/view descriptors include full root lineage; same-ubatch copy correlation uses exact root identity, not pointer equality across independent graphs.

An adapted existing external scheduler fixture passed42 forwarding/lifetime/failure cases. Its owner smoke checks actual allocated graph source/root coverage and CPU/GPU buffer ownership. All five native uncached8K requests, including warmups, have identical16-token arrays and draft counters to the preceding default cohort. Exit0, no warnings/floor breach, actual minimum1323.25MiB>=512. Runtime source/library hashes and target shard stat identities remain unchanged. These are observer validation checks, not full-model correctness gates for a new implementation.

| Same-process timing control | PP8K, t/s |
| --- | ---: |
| Unarmed pre | 1418.65 |
| Armed full metadata | 1173.11 |
| Unarmed post | 1422.22 |
| Mean unarmed | 1420.43 |

The armed request is17.41% below native control mean. Do not use its duration as engine speed or treat metadata collection as a pipeline gain. The controls still have the pass-through observer loaded; no new independent interposer-free A/B was run. TG16 is diagnostic only.

## Actual roots, writes and first readers

At full-tail prefix4096, all four CSA cache roots are Q8_0, shape[512,100352,1,1], each54591488bytes. They are persistent allocations outside per-copy input storage. Each current value block is F32[512,512,1,1],1048576bytes; tail508 has the corresponding508-row value shape. Numeric index values were not read.

| Persistent cache root | Actual device | Write graph | First non-view reader |
| --- | --- | ---: | --- |
| cache_k_l20 | CUDA2 | 26 | 30: k_all-20 CONCAT |
| cache_k_l23 | CUDA3 | 27 | 35: k_all-23 CONCAT |
| cache_k_l30 | CUDA4 | 28 | 40: k_all-30 CONCAT |
| cache_k_l38 | CUDA5 | 29 | 49: k_all-38 CONCAT |

The actual value producer is graph25 onCUDA4 (`comp_kv-20` ROPE, then MUL_MAT Hadamard with a local protected copy of `attn_inp_k_rot`). Direct peer-copy records correlate that root to the values used by each remote SET_ROWS. This corrects the assumption that the latent must physically originate on the source layer's GPU. The cached data themselves are already local to their readers; the candidate concerns when the remote writes are submitted, not removing those replicas.

The write at graph28 has a real CUDA3->CUDA4 transition wait before it. SET_ROWS is an in-place view of destination cache and consumes values, indices, destination in that order. It is not a no-op despite the tensor name. Graph nodes that follow the write may themselves be views into a different device's producer; one split's first/last names cannot establish all operand ownership.

At positions0 and2048 the CUDA4 row-index input copy has the same root buffer/data address and the same `#1` copy-slot suffix; intermediate captured names include`#2` and`#0`. The persistent cache root also remains stable. Thus required slot retirement guards actual reused storage. This does not prove arbitrary tensor ranges disjoint or authorize removing a fence. [Checks and exact lineage](artifacts/checks.json), [full summary](artifacts/summary.json).

## Existing mechanism and safe next scope

`LLAMA_DSV41_DEFER_CACHE_UPDATES` is default ON for decode/verification only. Its existing map queues remote CSA/LID writes until their reader layer. The eager prefill path first fans LID writes across devices; graph placement then leaves source20 rotation/Hadamard on the last LID destination, followed by a CSA fanout and a return to the source GPU.

A bounded next prototype can reuse the existing map for prefill, preserve all SET_ROWS/copy/event operations and perform writes immediately before their first local reader. It must handle KV-only separately: these chunks return at layer20 with n_outputs0 and an empty-map assertion. Every pending remote replica must be flushed after the source state/KV work and before return, even when no remote attention reader executes in that chunk. Removing the decode gate or empty-map assertion is not a correct implementation.

No source output pointer replacement, scheduler fence removal, new per-GPU worker, new transport format, or cross-graph pending queue is proposed. The [reviewable design](../../designs/1017-prefill-cache-deferral.md) starts OFF and specifies owner/lifetime admission, exact logits/taps/KV/state/rollback/NLL, KV-only/full-tail/slot-wrap/retry checks, actual memory and clean PP512/8K/98K/TG256 OFF/ON/OFF. Any remaining producer-placement change is a separate candidate. No speedup is promised by this audit.

## Maya comparison at source level

Maya chunk workers publish new compressed rows into bounded staging slots; a remote reader pulls into its local cache on its own stream before use, then acknowledges stage consumption. Source stage reuse waits for the old reader acknowledgements. Separate non-chunk code has an event-based replicate path. These are explicit lifetime mechanisms, not absence of synchronization. Our candidate borrows timing of reader-side updates while retaining our graph, Q8_0 quantization and existing copy/events. This audit did not run Maya again or quantify this mechanism's isolated gain.

[Source hashes and narrow snapshots](artifacts/source-audit.json) cover both runtimes. [Raw metadata, lossless gzip](artifacts/scopes.jsonl.gz) is verified against its original200612169bytes with SHA256 in [payload manifest](artifacts/payload.json). The raw JSONL and compiled helpers remain local/ignored; the compressed trace, source/build/controller and recovery-independent checks are preserved for review. Previous residual manifests remain unchanged.

## Next review boundary

The repository [AGENTS.md](../../../../AGENTS.md) requires: "If the change is large or introduces a new pattern, PAUSE and ask the user for confirmation". The proposed prefill ordering plus KV-only flush is such a pattern; the approved CPU lookup change was a different mechanism. Design confirmation is pending. Independent audit, cleanup and evidence validation are complete. GPUs are released; new runtime implementation/default promotion has not started.
