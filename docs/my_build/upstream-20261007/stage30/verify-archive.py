#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');DOC=REPO/'docs/my_build/upstream-20261007'
def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert head.startswith('9df5a4a40')
changed=subprocess.check_output(['git','diff','--name-only','--','src','ggml','common','tools'],cwd=REPO,text=True).splitlines();assert changed==['tools/server/server-context.cpp']
final=read(R/'final-summary.json');assert final['completed'] and final['total_responses']==66 and final['candidate_source_matches_repo'] and final['model_cuda_libraries_unchanged']
build=read(R/'preclear-build-manifest.json');assert sha(REPO/'tools/server/server-context.cpp')==build['source_sha256']==sha(build['source'])
for p,h in build['reused_objects'].items():assert sha(p)==h
assert all(sha(REPO/p)==h for p,h in build['base_sources'].items())
assert read(R/'diagnostic-controls.json')['narrow_fa_capture_accepted_as_baseline']
assert all(x['capture_equal'] and x['repeat_equal'] for x in read(R/'fa-replay-results/summary.json'))
requests={};manifests={};events={};count=0
for root in sorted((R/'server-runs').iterdir()):
 d=next(root.glob('glm5next-spec-*'));m=read(d/'manifest.json');meta=read(root/'control-manifest.json');result=read(d/'result.json');assert result['status']=='passed' and result['server_exit_code']==0
 hashes=meta['library_hashes'];assert m['head']==head
 for p,h in m['loaded_libraries'].items():assert sha(p)==h==hashes[Path(p).name]
 for p,h in meta['driver_sources'].items():assert sha(p)==h,(root,p)
 for p,st in meta['models'].items():assert Path(p).stat().st_size==st['size'] and Path(p).stat().st_mtime_ns==st['mtime_ns']
 for p,h in hashes.items():assert sha(Path(m['command'][0]).parent/p)==h
 for f in sorted(d.glob('*-request.json')):requests[root.name+'/'+f.name]=read(f)
 responses=list(d.glob('*-response.json'));count+=len(responses);assert len(result['requests'])==len(responses)
 for f in responses:
  v=read(f);assert len(v['tokens'])==v['tokens_predicted']==64 and v.get('truncated') is False
 manifests[root.name]={'control':meta,'server':m,'result':result}
 events[root.name]=[l for l in (d/'server.log').read_text().splitlines() if any(x in l for x in ['STAGE30_ROUNDTRIP','STAGE30_LAYOUT','STAGE30_TENSOR','restored context checkpoint','found better prompt','__TEST_TAG_CACHE_IDLE_SLOT__','GGML_ASSERT','cleaning up before exit'])]
assert count==126
for folder in ['cpu/before-native','cpu/occupied-before','cpu/state-before','gpu/tiny-before']:
 m=read(R/folder/'manifest.json');assert m['exit_code']==0
 source=Path(m.get('source',str(R/'ram-replay.cpp')));assert sha(source)==m['source_sha256']
 assert sha(m['library'])==m['library_sha256'];assert all(sha(R/folder/n)==h for n,h in m['outputs'].items())
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
(R/'source-final.patch').write_text(subprocess.check_output(['git','diff','--','tools/server/server-context.cpp'],cwd=REPO,text=True))
shutil.copy2(R/'fa-replay-results/summary.json',R/'fa-replay-summary.json')
raw={str(p.relative_to(R)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['raw-sha256.json','verification.json']}
save(R/'raw-sha256.json',raw)
verification={'head':head,'production_files_changed':changed,'source_matches_tested_candidate':True,'unchanged_model_cuda_snapshot':True,'server_processes':len(manifests),'diagnostic_responses':60,'candidate_responses':66,'loaded_libraries_verified':True,'driver_sources_verified':True,'model_stats_verified':True,'gpu_idle_after':True,'installed_server_replaced':False,'raw_index_sha256':sha(R/'raw-sha256.json')};save(R/'verification.json',verification)
out=DOC/'stage30';assert not out.exists();out.mkdir()
files=['README.md','run-server.py','run-final-server.py','run-cpu.py','run-gpu.py','ram-replay.cpp','ram-occupied-replay.cpp','ram-state-replay.cpp','ram-gpu-replay.cpp','make-trace.py','make-roundtrip.py','make-capture.py','make-fa-capture.py','make-preclear.py','build.py','build-context.py','build-server.py','analyze-responses.py','analyze-capture.py','analyze-diagnostics.py','analyze-final.py','prepare-fa.py','fa-replay.cpp','fa-replay-build-command.json','verify-archive.py','diagnostic-controls.json','fa-input-comparison.json','fa-replay-summary.json','final-summary.json','responses-compact.json','verification.json','raw-sha256.json','gpu-after.csv','source-final.patch']
files += [p.name for p in sorted(R.glob('*-build-manifest.json'))]
for name in files:shutil.copy2(R/name,out/name)
save(out/'requests.json',requests);save(out/'server-manifests.json',manifests);save(out/'selected-server-events.json',events)
for folder in ['cpu/before-native','cpu/occupied-before','cpu/state-before','gpu/tiny-before']:
 dst=out/folder;dst.mkdir(parents=True)
 for name in ['manifest.json','trace.jsonl']:shutil.copy2(R/folder/name,dst/name)
for root in sorted((R/'server-runs').iterdir()):
 if (root/'comparison.json').exists():
  dst=out/'comparisons';dst.mkdir(exist_ok=True);shutil.copy2(root/'comparison.json',dst/(root.name+'.json'))
save(out/'artifact-sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
shutil.copy2(R.parent/'STAGE30_REVIEW.md',DOC/'STAGE30_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/','](').replace('](llama_glm53_reasoning_budget/REVIEW.md)','](../glm53-reasoning-budget-20261008/REVIEW.md)'))
p=DOC/'README.md';s=p.read_text();s+='\n\n[Stage 30 report](STAGE30_REVIEW.md) localizes GLM5NEXT RAM-restore divergence to FlashAttention with a changed physical KV layout. The server now saves and clears other idle unified slots before RAM restore. Native, MTP and DFlash cache controls plus concurrent requests passed. The installed server is unchanged. Stop for review before committing the fix and archive.\n';p.write_text(s)
print(json.dumps(verification,indent=2))
