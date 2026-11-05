# First8K prefill raw forensic capture, 2026-11-05

Status: nine observed groups match; original divergence not reproduced and not declared fixed. Runtime source/defaults are unchanged after `d2d86c90f`; guarded MoE5+1 remains OFF by default. This cohort explicitly sets wide0 and prefill cache deferral0.

[Protocol](PROTOCOL.md), [harness](artifacts/prefill-forensic.cpp), [controller](artifacts/run.py), [raw run](artifacts/results.json), [analysis](artifacts/analysis.json), [summary](artifacts/summary.json), [blob manifest](artifacts/blob-manifest.json).

## Frozen scope and results

Three fresh processes each run three clear/reset repetitions of the original first8192-token technical-prose target/draft apply. Same private libllama `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3`, Q3 target/MXFP8 CPU Engram/Q8 KV/BF16-named mixed DSparkK5, ctx92160, batch8192/ubatch512, TopK2048, text/P1/threads12, six frozen Ampere/Ada UUIDs, fixed GPU overrides and15360 resident expert bundles. Actual draft SHA, target shard stat identities, module/source/harness hashes and expanded argv/env are recorded. No eval callback/profiler added; original unarmed fault interposer is retained.

The first observed group is the reference, without retrying to select a canonical match. All nine groups match that reference and the earlier archived first8K group, including taps37/38/39, target sequence/full and draft sequence/full state. Raw taps still match their original hashes after all state reads. Generic serializer descriptors for target/draft sequence/full states are metadata-consistent in every case.

| Fresh process | Reset repetitions | Exact vs first observed and canonical | Exit | Actual minimum free MiB |
|---|---:|---|---:|---:|
| 0 | 3 | yes | 0 | 1441.125 |
| 1 | 3 | yes | 0 | 1441.125 |
| 2 | 3 | yes | 0 | 1441.125 |

No recorded floor breach, OOM/CUDA error or compute-buffer mismatch warning. No faults were armed or injected; this cohort does not test failure recovery. 7 unique raw blobs (361579020 bytes) pass content-address rehash; binary payloads stay local/ignored, their manifest is versioned. There are no changed components to quantify in this cohort.

## Capture boundaries and interpretation

The original first8K read/hash order is retained: taps37/38/39, target sequence/full, draft sequence/full. Saved host vectors are retained and original CPU span pointers reused; only after all original reads/hashes are complete are raw blobs written and serializer metadata traversed over saved bytes, without new GPU reads. Retention changes host allocation lifetime and later reset timing. This cannot prove every original allocator/overlap history equivalent or guarantee universal determinism.

Read-only source checks show an immutable static MXFP8 LUT, lazy weights mapped through the ordinary CPU buffer/mmap path, and per-device/per-stream cuBLAS handles with explicit workspace. No fix to those components is justified by these observations alone; they do not establish a cause or exclude every scheduling/numerical/hardware issue.

The previous full-harness first8K and rollback failures remain preserved and unresolved. This cohort supplies raw canonical fixtures and a decoder for component/typed differences instead of hash-only evidence. A separate [full precision conditional capture](../1017-full-precision-forensic-20261105/PROTOCOL.md) repeats the original two-corpus sequence with no normal-path extra GPU reads/hashes and raw dumping only after a known failure point detects an actual difference. No default promotion, native speed or full two-corpus/NLL/failure acceptance is claimed here.
