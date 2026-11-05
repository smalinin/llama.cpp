from pathlib import Path
import json,hashlib
R=Path('/home/sergei/Github/llama.cpp');K=R/'docs/runtime-optimization';E=K/'experiments/1017-full-precision-forensic-20261105';P=E/'artifacts';j=json.loads((P/'precision-results.json').read_text());r=json.loads((P/'precision-1/results.json').read_text());a=j['arms'][0];assert j['complete']and r['complete']and a['groups_exact']==12 and len(r['faults'])==2 and not r['forensics'];refpath=Path(j['reference_results']['path']);ref={(x['corpus'],x['key']):x['result']for x in json.loads(refpath.read_text())['cases']};checks=[];nll=[]
for row in r['cases']:
 exact=row['result']==ref[row['corpus'],row['key']];assert exact;checks.append({'corpus':row['corpus'],'group':row['key'],'exact':exact})
 if row['key']=='nll128':nll.append({'corpus':row['corpus'],'tokens':len(row['result']['rows']),'mean_nll_reference':ref[row['corpus'],row['key']]['mean_nll'],'mean_nll_candidate':row['result']['mean_nll'],'delta_nll':row['result']['mean_nll']-ref[row['corpus'],row['key']]['mean_nll'],'query_width':4,'candidate_wide_branch_active_during_nll':False})
floor=min(x['min_free']/2**20 for phase in ['load','inference']for x in a['memory'][phase].values()if isinstance(x,dict)and'min_free'in x);assert floor>=512 and not a['floor_breach']and not a['warnings']and not a['memory'].get('error');summary={'complete':True,'status':'full_sequence_exact_with_conditional_capture','groups_exact':12,'corpora':2,'nll':nll,'faults_recovered':len(r['faults']),'injected_failures':a['faults']['injected_failures'],'conditional_captures':0,'actual_floor_mib':floor,'exit_code':a['exit_code'],'runtime_source_changed':False,'default_promoted':False,'original_failure_resolved':False,'native_speed_measured':False,'groups':checks};(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
report='''# Full precision sequence with conditional raw capture

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

'''
report+='| NLL corpus | Tokens | Reference | Candidate | Delta |\n|---|---:|---:|---:|---:|\n'
for x in nll:report+=f"| {x['corpus']} | {x['tokens']} | {x['mean_nll_reference']:.12f} | {x['mean_nll_candidate']:.12f} | 0 |\n"
report+='''
NLL is computed in the original4-query chunks. It guards the unchanged path; it does not independently exercise the new six-query branch. Six-query exactness is covered by the full vocabulary logits, restores/rollback here and by the separate [117-case shared-prefix diagnostic](../1017-verify-shared-prefix-20261105/REPORT.md). No generic tolerance was substituted for the exact comparisons.

Two deliberate seed-compute failures were injected, one per corpus. Both contexts clear to the exact empty state; both full-prefix retries recover the exact recorded first8K target/draft states. The expected failure log messages are part of this gate, not observed GPU crashes. The interposer records780scheduler compute calls,176injection graphs and2injected failures.

All comparisons complete; exit0. Actual sampled minimum: FLOOR MiB, above512 on every card. No recorded floor breach, unexpected OOM/CUDA allocation error, compute-buffer expectation warning or monitor error. Conditional capture entries:0; no changed raw payload was produced in this run. An external diagnostic compilation name collision was fixed before GPU launch; the original unexecuted build/source/log are retained.

## Decision and next step

This successful repeat adds full-model two-corpus precision/state/restore/rollback/failure evidence for5+1. It does not identify or fix the earlier first8K/rollback divergence. The unmodified original harness previously failed; neither its raw manifests nor exactness requirements are rewritten. Production defaults stay unchanged and no native TG/PP gain is claimed.

The next read-only performance stage can measure same-prefix widths1..6 and clean full-server OFF/ON/OFF, with identical prefix/output/draft-counter controls. If those controls diverge, retain the arm and do not use it as an equivalent speed comparison; use conditional raw capture to localize the difference. Full prefilling variability remains an open review item for default promotion. Floating sample-grid6 and weight-sharing kernels remain separate unstarted candidates.
'''.replace('FLOOR',str(floor));(E/'REPORT.md').write_text(report)
# Add current observations without changing historical data/manifests.
p=K/'README.md';p.write_text(p.read_text()+'''\n\n## Raw first8K and complete precision repeat\n\n[Nine OFF raw captures](experiments/1017-prefill-forensic-20261105/REPORT.md), three fresh processes, all match canonical taps/target/draft sequence/full state; metadata and saved raw taps remain consistent. The [full conditional-capture repeat](experiments/1017-full-precision-forensic-20261105/REPORT.md) passes all12 text/C++ groups, full/partial/sequence restore, rollback, append transitions, NLLdelta0 and two injected failure/retries; capture count0, actual floor1387.125MiB, exit0. NLL uses the old4-query chunks; six-query evidence is the full logits/state/rollback and shared-prefix matrix. Earlier uninstrumented failures remain unresolved and immutable. No new runtime/default or native speed change. Next: controlled same-prefix cost and native OFF/ON/OFF with exact output controls; reject mismatched arms rather than hiding prefill variability.\n''')
p=K/'designs/1017-six-query-moe-float.md';p.write_text(p.read_text()+'''\n## Full conditional repeat after raw first8K controls\n\n[Nine raw OFF groups](../experiments/1017-prefill-forensic-20261105/REPORT.md) and the [complete two-corpus repeat](../experiments/1017-full-precision-forensic-20261105/REPORT.md) now pass the observed exact gates, including two failures/retry. No conditional capture triggered. Old failures are not declared fixed; the diagnostic executable changes host allocation history. Original NLL4 does not exercise the new wide branch, which has separate exact full-logit/state/rollback checks. Next is read-only same-prefix/native cost with strict matching controls, while retaining full-prefill reproducibility as an open promotion item. Default stays OFF; Phase B/C remain unstarted.\n''')
p=K/'LESSONS.md';p.write_text(p.read_text()+'''\n\n## Capture after the original read/hash boundary, and state which branch a metric exercises\n\n[Nine first8K raw controls](experiments/1017-prefill-forensic-20261105/REPORT.md) and a [complete two-corpus conditional-capture repeat](experiments/1017-full-precision-forensic-20261105/REPORT.md) pass, but do not erase earlier uninstrumented failures. Keeping the original GPU read/hash order and dumping only after a mismatch avoids normal-path extra GPU reads; retaining host vectors and changing JSON/executable layout can still perturb allocator/timing histories. Record those limits.\n\nThe old NLL128 loop uses4-query chunks and cannot independently validate a selector active only at6queries. Report NLLdelta0 with its width and use the actual six-query full-vocabulary logits/state/rollback and dispatch evidence for the proposed mechanism. Matching repeated runs is observed evidence, not proof of universal baseline determinism or a root-cause fix.\n''')
p=K/'experiments/1017-verify-moe5plus1-20261105/REPORT.md';p.write_text(p.read_text()+'''\n## Subsequent raw controls and full conditional repeat\n\n[Nine fresh/reset OFF first8K captures](../1017-prefill-forensic-20261105/REPORT.md) match exactly. The [full two-corpus conditional-capture repeat](../1017-full-precision-forensic-20261105/REPORT.md) passes12groups, all restores/rollback/append transitions, original NLL4delta0 and two seed failure/retries; raw capture count0, minimum1387.125MiB, exit0. Original uninstrumented failures remain unresolved; the diagnostic layout differs and no exactness requirement is relaxed. No new default/runtime or speed result. Next: same-prefix cost and native OFF/ON/OFF with matching controls.\n''')
print('complete',floor,'NLL',nll)
