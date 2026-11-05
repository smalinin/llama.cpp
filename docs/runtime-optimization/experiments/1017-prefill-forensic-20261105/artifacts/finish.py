from pathlib import Path
import json,hashlib
R=Path('/home/sergei/Github/llama.cpp');K=R/'docs/runtime-optimization';E=K/'experiments/1017-prefill-forensic-20261105';P=E/'artifacts';j=json.loads((P/'results.json').read_text());assert j['complete'];analysis=json.loads((P/'analysis.json').read_text());assert len(analysis['cases'])==9
manifest=[]
for p in sorted((P/'bundle/blobs').glob('*.bin')):
 with p.open('rb')as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==p.stem;manifest.append({'file':p.name,'bytes':p.stat().st_size,'sha256':h})
(P/'blob-manifest.json').write_text(json.dumps({'all_verified':True,'count':len(manifest),'total_bytes':sum(x['bytes']for x in manifest),'files':manifest},indent=2)+'\n');arms=[]
for a in j['arms']:
 floor=min(x['min_free']/2**20 for phase in ['load','inference']for x in a['memory'][phase].values()if isinstance(x,dict)and'min_free'in x);arms.append({'index':a['index'],'cases':3,'reference_exact':a['reference_exact'],'canonical_exact':a['canonical_exact'],'metadata_consistent':a['metadata_consistent'],'tap_raw_matches_original':a['tap_raw_matches_original'],'exit':a['exit_code'],'floor_mib':floor})
summary={'complete':True,'status':'original_divergence_not_reproduced','groups':9,'fresh_processes':3,'all_reference_exact':j['all_reference_exact'],'all_canonical_exact':j['all_canonical_exact'],'all_metadata_consistent':all(all(a['metadata_consistent'])for a in arms),'all_raw_taps_match':all(all(a['tap_raw_matches_original'])for a in arms),'runtime_changed':False,'default_promoted':False,'original_failure_resolved':False,'floor_mib':min(a['floor_mib']for a in arms),'arms':arms};(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
s='''# First8K prefill raw forensic capture, 2026-11-05

Status: nine observed groups match; original divergence not reproduced and not declared fixed. Runtime source/defaults are unchanged after `d2d86c90f`; guarded MoE5+1 remains OFF by default. This cohort explicitly sets wide0 and prefill cache deferral0.

[Protocol](PROTOCOL.md), [harness](artifacts/prefill-forensic.cpp), [controller](artifacts/run.py), [raw run](artifacts/results.json), [analysis](artifacts/analysis.json), [summary](artifacts/summary.json), [blob manifest](artifacts/blob-manifest.json).

## Frozen scope and results

Three fresh processes each run three clear/reset repetitions of the original first8192-token technical-prose target/draft apply. Same private libllama `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3`, Q3 target/MXFP8 CPU Engram/Q8 KV/BF16-named mixed DSparkK5, ctx92160, batch8192/ubatch512, TopK2048, text/P1/threads12, six frozen Ampere/Ada UUIDs, fixed GPU overrides and15360 resident expert bundles. Actual draft SHA, target shard stat identities, module/source/harness hashes and expanded argv/env are recorded. No eval callback/profiler added; original unarmed fault interposer is retained.

The first observed group is the reference, without retrying to select a canonical match. All nine groups match that reference and the earlier archived first8K group, including taps37/38/39, target sequence/full and draft sequence/full state. Raw taps still match their original hashes after all state reads. Generic serializer descriptors for target/draft sequence/full states are metadata-consistent in every case.

| Fresh process | Reset repetitions | Exact vs first observed and canonical | Exit | Actual minimum free MiB |
|---|---:|---|---:|---:|
'''
for a in arms:s+=f"| {a['index']} | 3 | yes | {a['exit']} | {a['floor_mib']} |\n"
s+='''
No recorded floor breach, OOM/CUDA error or compute-buffer mismatch warning. No faults were armed or injected; this cohort does not test failure recovery. CONTENT_COUNT unique raw blobs (CONTENT_BYTES bytes) pass content-address rehash; binary payloads stay local/ignored, their manifest is versioned. There are no changed components to quantify in this cohort.

## Capture boundaries and interpretation

The original first8K read/hash order is retained: taps37/38/39, target sequence/full, draft sequence/full. Saved host vectors are retained and original CPU span pointers reused; only after all original reads/hashes are complete are raw blobs written and serializer metadata traversed over saved bytes, without new GPU reads. Retention changes host allocation lifetime and later reset timing. This cannot prove every original allocator/overlap history equivalent or guarantee universal determinism.

Read-only source checks show an immutable static MXFP8 LUT, lazy weights mapped through the ordinary CPU buffer/mmap path, and per-device/per-stream cuBLAS handles with explicit workspace. No fix to those components is justified by these observations alone; they do not establish a cause or exclude every scheduling/numerical/hardware issue.

The previous full-harness first8K and rollback failures remain preserved and unresolved. This cohort supplies raw canonical fixtures and a decoder for component/typed differences instead of hash-only evidence. A separate [full precision conditional capture](../1017-full-precision-forensic-20261105/PROTOCOL.md) repeats the original two-corpus sequence with no normal-path extra GPU reads/hashes and raw dumping only after a known failure point detects an actual difference. No default promotion, native speed or full two-corpus/NLL/failure acceptance is claimed here.
''';s=s.replace('CONTENT_COUNT',str(len(manifest))).replace('CONTENT_BYTES',str(sum(x['bytes']for x in manifest)));(E/'REPORT.md').write_text(s);print(summary['floor_mib'],len(manifest),sum(x['bytes']for x in manifest))
