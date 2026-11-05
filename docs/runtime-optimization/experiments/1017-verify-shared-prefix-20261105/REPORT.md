# DeepSeek41 verification from common captured prefix state

Status: completed diagnostic PASS; original full-prefill discrepancy remains unresolved. Runtime source and production modules are unchanged after commit `e69c07128`. `LLAMA_DSV41_VERIFY_MOE_WIDE` remains opt-in/default OFF. This does not erase the failures in the [Phase A report](../1017-verify-moe5plus1-20261105/REPORT.md).

## Protocol and identity

[Protocol](PROTOCOL.md), [controller](artifacts/run.py), [harness source](artifacts/shared-prefix.cpp), [build](artifacts/build.json), [raw manifest](artifacts/results.json), [analysis](artifacts/analysis.json), [summary](artifacts/summary.json).

Same private libllama SHA256 `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3`, unchanged target/kernel/common modules, six fixed Ampere/Ada UUIDs, Q3_K target with Q4_K down experts0..4, MXFP8 CPU Engram, Q8 KV, ctx92160, batch8192/ubatch512, TopK2048, P1/threads12, full15360 GPU expert bundles, BF16-named mixed DSparkK5, raw target/draft logits and p-min0. Prefill cache deferral stays0. The actual [sidecar SHA](artifacts/draft-identity.json) matches `4f8bcd87f47f106e2f7fb24314b415462558ebb6d15f01c77e2ea5a00c1aba44`; target shard stat identities and module hashes are checked before every arm.

First OFF captures target/draft full and sequence state after the original90112-token prose prefill, fixed prompt IDs and anchor. ON and final OFF restore exactly those host blobs without recomputing prefill. Full and sequence roundtrips pass; every verification begins from byte-exact restored target/draft prefix states. Standalone K3/K5 drafts validate readiness of the restored injected draft KV independently of retained target features. The seed matches the archived first8K states/taps, final target/draft sequence states and last logits; it was not selected by retrying for an old hash. See [reference checks](artifacts/reference-checks.json).

## Results

| Startup selector | Context history | Recorded target cases | Standalone drafts | Exact vs first OFF | Actual minimum free MiB | Exit |
|---|---|---:|---:|---|---:|---:|
| 0 | prefill, capture, restore | 39 | 6 | yes | 1443.125 | 0 |
| 1 | restore shared blobs | 39 | 6 | yes | 1511.125 | 0 |
| 0 | restore shared blobs | 39 | 6 | yes | 1507.125 | 0 |

All117 target cases and18 standalone draft checks match. Three waves per arm cover widths1/3/4/5/6, repeated width5/6, shape/slot transitions and rollback4/keep1,2,4 plus6/keep1,3,6. Target/draft state, target logits/taps, verified proposal IDs and standalone draft IDs/confidence match across arms. Every arm passes its internal next-draft and target-state replay comparisons; repeated waves within each arm also match. No new NLL, partial-state, fault-injection or two-corpus full-model gate is claimed.

Actual dispatch is confirmed for all15 six-query main calls per arm:

| Logical backend counter per six-query target call | OFF | ON5+1 | final OFF |
|---|---:|---:|---:|
| MoE MUL_MAT_ID nodes | 720 | 240 | 720 |
| Floating MUL_MAT nodes | 950 | 950 | 950 |
| Backend graph-compute submissions | 113 | 57 | 113 |

The existing counters interposer observes target decode/sync only. These are diagnostic logical operations/backend graph calls, not physical kernel totals or native TG. Cold/reused CUDA graph instantiation counts remain in each raw row; no timing comparison was made. Phase B floating batching is inactive.

The earlier joint admission is reused only because source/context/output capacity/placement and all GPU tensor requirements are unchanged; actual monitored floors exceed512MiB throughout. All arms exit0, with no recorded floor breach, OOM/CUDA allocation errors or compute-buffer mismatch warning. The minimum is1443.125MiB. Raw blobs are local ignored artifacts; all 29 content-addressed files (1811852816 bytes) pass rehash, with a versioned [manifest](artifacts/blob-manifest.json). A controller syntax typo and missing optional NumPy were corrected before their respective GPU launch/analysis, without changing runtime or the executing test. Initial diagnostic sources are retained; final analysis uses Python's standard library.

## Interpretation and remaining work

On this common prefix and checked widths/rollback matrix,5+1 has the same observed numerical/state behavior as the old path and reduces MoE dispatch fragmentation. This supports the bounded mechanism; it does not explain the initial full-harness rollback failure or the later first8K prefill mismatch. Restore arms deliberately have different prefill/allocator history. Their success cannot prove that all original timing/lifetime histories are safe, or quantify any quality/performance impact of the earlier divergence.

Default stays OFF. Before adoption, capture the first divergent full-prefill output/state without changing work before that operation, separating independent startup/prefill variation from later verification. Then close the original two-corpus precision/NLL/fault and native OFF/ON/OFF gates. [Next forensic protocol](NEXT_PREFILL_FORENSIC.md) is prepared; no new runtime synchronization/default change or speculative diagnosis has been adopted.
