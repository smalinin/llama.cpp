# Full precision sequence with conditional raw capture

Status: this complete repeat passes exactly; the earlier failures remain unresolved and preserved. Runtime/defaults are unchanged after commit `d2d86c90f`; `LLAMA_DSV41_VERIFY_MOE_WIDE` remains opt-in/OFF. This run explicitly uses1 for the existing5+1 prototype.

[Protocol](PROTOCOL.md), [source](artifacts/full-forensic.cpp), [build](artifacts/build.json), [controller](artifacts/run.py), [raw manifest](artifacts/precision-results.json), [raw outputs](artifacts/precision-1/results.json), [summary](artifacts/summary.json).

## Scope and capture boundary

Same private libllama `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3` and frozen Q3 target/MXFP8 CPU Engram/Q8 KV/DSparkK5/text/P1/new-host GPU layout as Phase A. Context92160, batch8192/ubatch512, TopK2048, all15360 experts on GPU, prefill cache deferral0, raw target/draft logits and p-min0. The controller verifies the runtime source/module hashes against the completed [nine first8K OFF controls](../1017-prefill-forensic-20261105/REPORT.md); no production module is overwritten.

Reuse the original two-corpus precision sequence and unarmed/armed fault interposer. Conditional raw dumping at the known first8K and rollback mismatch points occurs only after the original hash/comparison detects a difference. No normal-path extra GPU reads or hashes are added. Target verify/tail hashes precede the two final target state reads/hashes in the original order. Replay vectors and JSON objects have different host allocation lifetimes; the new executable is not asserted to reproduce every original timing/allocator history. It is not the byte-identical uninstrumented harness, nor a native speed test.

## Observed gates

All12 result groups match the frozen OFF reference exactly:

| Group | Technical prose | C++ source |
|---|---|---|
| First8K, checkpoints and90112-token prefix | exact | exact |
| Standalone K3/K5 draft IDs/confidence/logits/state | exact | exact |
| Full/sequence/partial restore, checkpoints and80K resume | exact | exact |
| Rejected/full rollback, logits and next-draft replay/state | exact | exact |
| Append widths1/7/508/511/512/513 | exact | exact |
|128teacher-forced NLL rows and final state | exact | exact |

| NLL corpus | Tokens | Reference | Candidate | Delta |
|---|---:|---:|---:|---:|
| technical-prose | 128 | 0.372954987845 | 0.372954987845 | 0 |
| cpp-source | 128 | 0.105268773228 | 0.105268773228 | 0 |

NLL is computed in the original4-query chunks. It guards the unchanged path; it does not independently exercise the new six-query branch. Six-query exactness is covered by the full vocabulary logits, restores/rollback here and by the separate [117-case shared-prefix diagnostic](../1017-verify-shared-prefix-20261105/REPORT.md). No generic tolerance was substituted for the exact comparisons.

Two deliberate seed-compute failures were injected, one per corpus. Both contexts clear to the exact empty state; both full-prefix retries recover the exact recorded first8K target/draft states. The expected failure log messages are part of this gate, not observed GPU crashes. The interposer records780scheduler compute calls,176injection graphs and2injected failures.

All comparisons complete; exit0. Actual sampled minimum: 1387.125 MiB, above512 on every card. No recorded floor breach, unexpected OOM/CUDA allocation error, compute-buffer expectation warning or monitor error. Conditional capture entries:0; no changed raw payload was produced in this run. An external diagnostic compilation name collision was fixed before GPU launch; the original unexecuted build/source/log are retained.

## Decision and next step

This successful repeat adds full-model two-corpus precision/state/restore/rollback/failure evidence for5+1. It does not identify or fix the earlier first8K/rollback divergence. The unmodified original harness previously failed; neither its raw manifests nor exactness requirements are rewritten. Production defaults stay unchanged and no native TG/PP gain is claimed.

The next read-only performance stage can measure same-prefix widths1..6 and clean full-server OFF/ON/OFF, with identical prefix/output/draft-counter controls. If those controls diverge, retain the arm and do not use it as an equivalent speed comparison; use conditional raw capture to localize the difference. Full prefilling variability remains an open review item for default promotion. Floating sample-grid6 and weight-sharing kernels remain separate unstarted candidates.
