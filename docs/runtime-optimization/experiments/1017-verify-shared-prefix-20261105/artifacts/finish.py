from pathlib import Path
import json,hashlib,collections
R=Path('/home/sergei/Github/llama.cpp');K=R/'docs/runtime-optimization';E=K/'experiments/1017-verify-shared-prefix-20261105';P=E/'artifacts';run=json.loads((P/'results.json').read_text());analysis=json.loads((P/'analysis.json').read_text());assert run['complete']and run['all_internal_exact']and run['all_cross_arm_exact'];assert all(a['complete']and a['cases']==39 and a['within_arm_repeats_exact']and not a['changed_cases']for a in analysis['arms']);
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=[]
for p in sorted((P/'bundle/blobs').glob('*.bin')):
 h=sha(p);assert h==p.stem;manifest.append({'file':p.name,'bytes':p.stat().st_size,'sha256':h})
(P/'blob-manifest.json').write_text(json.dumps({'all_content_addresses_verified':True,'count':len(manifest),'total_bytes':sum(x['bytes']for x in manifest),'files':manifest},indent=2)+'\n')
old=json.loads((K/'experiments/1017-prefill-cache-deferral-20261105/artifacts/precision-0/results.json').read_text());oldprefix=next(x['result']for x in old['cases']if x['corpus']=='technical-prose'and x['key']=='prefix90112');seed=json.loads((P/'bundle/seed.json').read_text());old_drafts=next(x['result']for x in old['cases']if x['corpus']=='technical-prose'and x['key']=='draftK3K5');r0=json.loads((P/'arm-0-0/results.json').read_text());checks={'seed_first8K_matches_archived':seed['first8K']==oldprefix['first8K'],'seed_target_seq_matches_archived':seed['target_seq']==oldprefix['target_seq'],'seed_draft_seq_matches_archived':seed['draft_seq']==oldprefix['draft_seq'],'seed_last_logits_matches_archived':seed['last_logits']==oldprefix['last_logits'],'off_standalone_drafts_match_archived':all(d['result']==old_drafts[[3,5].index(d['k'])]for d in r0['draft_checks'])};assert all(checks.values());(P/'reference-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
rows=[];inst={}
for arm in run['arms']:
 res=json.loads((P/f"arm-{arm['index']}-{arm['flag']}/results.json").read_text());n6=[x for x in res['cases']if x['value']['n']==6];n6counts={k:sorted(set(x['counts'][k]for x in n6))for k in ['moe_matmul','float_matmul','graphs','cuda_graph_instantiate']};floor=min(x['min_free']/2**20 for phase in ['load','inference']for x in arm['memory'][phase].values()if isinstance(x,dict)and'min_free'in x);rows.append({'index':arm['index'],'flag':arm['flag'],'mode':arm['mode'],'cases':len(res['cases']),'draft_checks':len(res['draft_checks']),'floor_mib':floor,'exit_code':arm['exit_code'],'counts':n6counts});inst[str(arm['index'])]=n6counts['cuda_graph_instantiate']
summary={'complete':True,'scope':'Common frozen90112-token prose prefix, raw Q3/Q8/DSparkK5 text/P1 diagnostic; source/modules unchanged aftere69c07128. StartupOFF/ON/OFF, same target/draft full and sequence blobs. Not full native/NLL/failure acceptance.','runtime_commit':run['head'],'cases_total':117,'standalone_draft_checks_total':18,'all_cases_exact':True,'within_arm_repeats_exact':True,'activation_proven':True,'default_promoted':False,'native_speed_measured':False,'original_failure_resolved':False,'blob_integrity':True,'arms':rows,'reference_checks':checks};(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
report='''# DeepSeek41 verification from common captured prefix state

Status: completed diagnostic PASS; original full-prefill discrepancy remains unresolved. Runtime source and production modules are unchanged after commit `e69c07128`. `LLAMA_DSV41_VERIFY_MOE_WIDE` remains opt-in/default OFF. This does not erase the failures in the [Phase A report](../1017-verify-moe5plus1-20261105/REPORT.md).

## Protocol and identity

[Protocol](PROTOCOL.md), [controller](artifacts/run.py), [harness source](artifacts/shared-prefix.cpp), [build](artifacts/build.json), [raw manifest](artifacts/results.json), [analysis](artifacts/analysis.json), [summary](artifacts/summary.json).

Same private libllama SHA256 `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3`, unchanged target/kernel/common modules, six fixed Ampere/Ada UUIDs, Q3_K target with Q4_K down experts0..4, MXFP8 CPU Engram, Q8 KV, ctx92160, batch8192/ubatch512, TopK2048, P1/threads12, full15360 GPU expert bundles, BF16-named mixed DSparkK5, raw target/draft logits and p-min0. Prefill cache deferral stays0. The actual [sidecar SHA](artifacts/draft-identity.json) matches `4f8bcd87f47f106e2f7fb24314b415462558ebb6d15f01c77e2ea5a00c1aba44`; target shard stat identities and module hashes are checked before every arm.

First OFF captures target/draft full and sequence state after the original90112-token prose prefill, fixed prompt IDs and anchor. ON and final OFF restore exactly those host blobs without recomputing prefill. Full and sequence roundtrips pass; every verification begins from byte-exact restored target/draft prefix states. Standalone K3/K5 drafts validate readiness of the restored injected draft KV independently of retained target features. The seed matches the archived first8K states/taps, final target/draft sequence states and last logits; it was not selected by retrying for an old hash. See [reference checks](artifacts/reference-checks.json).

## Results

| Startup selector | Context history | Recorded target cases | Standalone drafts | Exact vs first OFF | Actual minimum free MiB | Exit |
|---|---|---:|---:|---|---:|---:|
'''
for row in rows:report+=f"| {row['flag']} | {'prefill, capture, restore'if row['index']==0 else'restore shared blobs'} | 39 | 6 | yes | {row['floor_mib']} | {row['exit_code']} |\n"
report+='''
All117 target cases and18 standalone draft checks match. Three waves per arm cover widths1/3/4/5/6, repeated width5/6, shape/slot transitions and rollback4/keep1,2,4 plus6/keep1,3,6. Target/draft state, target logits/taps, verified proposal IDs and standalone draft IDs/confidence match across arms. Every arm passes its internal next-draft and target-state replay comparisons; repeated waves within each arm also match. No new NLL, partial-state, fault-injection or two-corpus full-model gate is claimed.

Actual dispatch is confirmed for all15 six-query main calls per arm:

| Logical backend counter per six-query target call | OFF | ON5+1 | final OFF |
|---|---:|---:|---:|
| MoE MUL_MAT_ID nodes | 720 | 240 | 720 |
| Floating MUL_MAT nodes | 950 | 950 | 950 |
| Backend graph-compute submissions | 113 | 57 | 113 |

The existing counters interposer observes target decode/sync only. These are diagnostic logical operations/backend graph calls, not physical kernel totals or native TG. Cold/reused CUDA graph instantiation counts remain in each raw row; no timing comparison was made. Phase B floating batching is inactive.

The earlier joint admission is reused only because source/context/output capacity/placement and all GPU tensor requirements are unchanged; actual monitored floors exceed512MiB throughout. All arms exit0, with no recorded floor breach, OOM/CUDA allocation errors or compute-buffer mismatch warning. The minimum is1443.125MiB. Raw blobs are local ignored artifacts; all CONTENT_COUNT content-addressed files (CONTENT_BYTES bytes) pass rehash, with a versioned [manifest](artifacts/blob-manifest.json). A controller syntax typo and missing optional NumPy were corrected before their respective GPU launch/analysis, without changing runtime or the executing test. Initial diagnostic sources are retained; final analysis uses Python's standard library.

## Interpretation and remaining work

On this common prefix and checked widths/rollback matrix,5+1 has the same observed numerical/state behavior as the old path and reduces MoE dispatch fragmentation. This supports the bounded mechanism; it does not explain the initial full-harness rollback failure or the later first8K prefill mismatch. Restore arms deliberately have different prefill/allocator history. Their success cannot prove that all original timing/lifetime histories are safe, or quantify any quality/performance impact of the earlier divergence.

Default stays OFF. Before adoption, capture the first divergent full-prefill output/state without changing work before that operation, separating independent startup/prefill variation from later verification. Then close the original two-corpus precision/NLL/fault and native OFF/ON/OFF gates. [Next forensic protocol](NEXT_PREFILL_FORENSIC.md) is prepared; no new runtime synchronization/default change or speculative diagnosis has been adopted.
''';report=report.replace('CONTENT_COUNT',str(len(manifest))).replace('CONTENT_BYTES',str(sum(x['bytes']for x in manifest)));(E/'REPORT.md').write_text(report)
(E/'NEXT_PREFILL_FORENSIC.md').write_text('''# Next: capture full-prefill variability before changing runtime

Reuse the original raw precision harness setup with ctx92160/Q8/Q3/DSparkK5 and the original unarmed fault interposer, MoE-wide0 and prefill cache deferral0. No eval callbacks or new GPU reads before completion of the first8K target+draft apply. The new six-query branch cannot activate during that prefix.

Retain the existing first8K capture order: taps37/38/39, target sequence/full, draft sequence/full. Preserve those host bytes instead of discarding them after hashing. Only after every original read/hash completes, write content-addressed raw blobs and compare the complete group. Extra forensic CPU/disk work occurs after the candidate failure point. Do not silently normalize uninitialized state, NaNs, signed zeros or metadata.

First fresh process captures the reference without choosing a seed by its canonical hash. Record whether it matches the archived group. Run three clear/reset repeats per process, then fresh startup processes with the same frozen identities. Localize first changed tap/serializer component and quantify raw differences. A passed same-process repeat does not replace a fresh-start control. Retain failures and do not call a valid numerical backend difference a hardware fault without evidence.

After a reproducible divergence is captured, isolate one supported scheduling/graph/numerical mechanism at a time. No new runtime ownership/order change follows from this protocol; that needs a separate reviewable design if required. The current source already uses per-device/per-stream cuBLAS handles and explicit workspace; do not assume a shared cuBLAS handle is the cause. Do not remove scheduler fences or relax exactness based on hash-only symptoms.

Once prefill variability is understood or controlled, rerun full two-corpus/NLL/fault gates for5+1, followed by native same-K OFF/ON/OFF speed. Phase B/C remain separate candidates, not combined into the forensic cohort.
''')
# Historical raw manifests are untouched; append new evidence to narrative indexes.
p=K/'experiments/1017-verify-moe5plus1-20261105/REPORT.md';p.write_text(p.read_text()+'''\n## Subsequent shared-prefix isolation\n\n[Completed OFF/ON/OFF diagnostic](../1017-verify-shared-prefix-20261105/REPORT.md), after commit e69c07128: all117 target cases and18 standalone K3/K5 checks match from the same captured90112-token target/draft prefix. All three waves/rollback replays are exact. Actual six-query MoE nodes720->240->720 and backend submissions113->57->113; floating nodes950 unchanged. Minimum free1443.125MiB, exits0. This isolates the checked verification mechanism from independent prefill values; original full-prefill/rollback failure attribution remains unresolved. No native speed, NLL completion or default promotion. Next: first8K raw forensic capture before changing runtime.\n''')
p=K/'designs/1017-six-query-moe-float.md';p.write_text(p.read_text()+'''\n## Shared-prefix gate, after e69c07128\n\n[OFF/ON/OFF isolation](../experiments/1017-verify-shared-prefix-20261105/REPORT.md) passes117 target cases/18draft checks, exact restored prefix and repeated state/logit/rollback comparisons. MoE720->240 and backend submissions113->57; floating950 unchanged. This is a diagnostic, not a native speed/full acceptance result. Original failures remain unresolved; the next bounded task is raw first8K prefill forensics with no added work before the failure point. Defaults stay OFF; Phase B/C have not started.\n''')
p=K/'LESSONS.md';p.write_text(p.read_text()+'''\n\n## Freeze prefix values before attributing a verification difference\n\nThe [shared-prefix diagnostic](experiments/1017-verify-shared-prefix-20261105/REPORT.md) restores the same target/draft state in fresh startupOFF/ON/OFF contexts.117target cases and18draft checks match; actual MoE grouping reduces logical nodes720->240 without a floating-path change. Independent prefill variability must not be mistaken for candidate arithmetic drift. But shared-state success also cannot erase an earlier full-run failure: allocator/graph history differs, and full prefill/NLL/fault/native gates remain separate. Preserve raw changed host blobs only after the original failing operation completes to avoid perturbing the operation being diagnosed.\n''')
p=K/'README.md';p.write_text(p.read_text()+'''\n\n## Shared-prefix MoE5+1 isolation: diagnostic PASS\n\n[New OFF/ON/OFF evidence](experiments/1017-verify-shared-prefix-20261105/REPORT.md): same saved90112-token target/draft state,117target cases/18standalone drafts, exact logits/taps/state/rollback across all arms and repeats. Actual MoE nodes720->240->720, backend graph calls113->57->113, floating950 unchanged; actual floor1443.125MiB, exits0. These are counters, not a speed claim. The original full-harness failures are unresolved; next is first8K raw forensics, then full two-corpus/NLL/fault/native gates. Flag remains OFF and production modules unchanged.\n''')
print('BLOBS',len(manifest),sum(x['bytes']for x in manifest));print(rows)
