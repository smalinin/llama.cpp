#!/usr/bin/env python3
"""Verify test provenance and write a compact review archive."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');PREV=R.parent/'stage28';DOC=REPO/'docs/my_build/upstream-20261007'
def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert head.startswith('5a4de37b8')
assert not subprocess.check_output(['git','diff','--name-only','--','src','common','ggml','tools'],cwd=REPO,text=True).strip()
b=read(PREV/'final-build-manifest.json');assert all(sha(REPO/n)==h for n,h in b['sources'].items())
assert all(sha(PREV/'final-bin'/n)==h for n,h in b['candidate_hashes'].items())
cpu=read(R/'cpu-manifest.json');assert sha(cpu['model'])==cpu['model_sha256'] and sha(cpu['library'])==cpu['library_sha256']==b['candidate_hashes']['libllama.so.0.4.0']
assert sha(R/'restore-replay.cpp')==cpu['source_sha256'] and sha(R/'restore-replay')==cpu['binary_sha256']
for label,v in cpu['cases'].items():
 assert v['exit_code']==0 and all(sha(R/'cpu'/label/n)==h for n,h in v['float_sha256'].items())
s=read(R/'summary.json');assert len(s['server'])>=4 and all(v['status']=='passed' for v in s['server'].values())
raw_requests={};raw_responses={};manifests={};selected={}
for control in sorted((R/'server-runs').iterdir()):
 d=next(control.glob('glm5next-spec-*'));meta=read(control/'control-manifest.json');m=read(d/'manifest.json');result=read(d/'result.json')
 assert m['head']==head and result['server_exit_code']==0 and len(result['requests'])==(12 if '-ram' in control.name else 16)
 assert meta['source_hashes']==b['sources'] and meta['library_hashes']==b['candidate_hashes']
 assert all(sha(p)==h for p,h in meta['driver_sources'].items())
 for p,h in m['loaded_libraries'].items():assert Path(p).parent==PREV/'final-bin' and sha(p)==h==b['candidate_hashes'][Path(p).name]
 for p,st in meta['models'].items():assert Path(p).stat().st_size==st['size'] and Path(p).stat().st_mtime_ns==st['mtime_ns']
 for p in sorted(d.glob('*-request.json')):raw_requests[control.name+'/'+p.name]=read(p)
 for p in sorted(d.glob('*-response.json')):
  v=read(p);assert len(v['tokens'])==v['tokens_predicted']==64 and v.get('truncated') is False
  assert len(v['completion_probabilities'])==64 and len(v['completion_probabilities'][0]['top_logprobs'])==5
  assert all(len(x['top_logprobs']) in [0,5] for x in v['completion_probabilities'])
  raw_responses[control.name+'/'+p.name]={k:v.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated','tokens_evaluated','tokens_predicted','timings']}
  raw_responses[control.name+'/'+p.name]['raw_response_sha256']=sha(p)
  raw_responses[control.name+'/'+p.name]['probabilities_sha256']=hashlib.sha256(json.dumps(v['completion_probabilities'],sort_keys=True).encode()).hexdigest()
  raw_responses[control.name+'/'+p.name]['nonempty_probability_rows']={str(i):row for i,row in enumerate(v['completion_probabilities']) if row['top_logprobs']}
 manifests[control.name]={'control':meta,'server':m,'result':result}
 selected[control.name]=[l.rstrip() for l in (d/'server.log').read_text().splitlines() if any(x in l for x in ['restored context checkpoint','failed to remove','unable to restore','restoring prompt','found better prompt','GGML_ASSERT','cleaning up before exit'])]
 if '-ram' in control.name:
  hits=[q for q in result['requests'] if q['phase'].startswith('reuse') and q['timings'].get('cache_n',0)>0]
  assert len(hits)==6 and all(len(set(ids))==1 for ids in s['server'][control.name]['slots_per_prompt'].values())
  if control.name=='native-ram-off':assert s['server'][control.name]['probability_coverage']['nonempty_top5']==768
  assert sum('found better prompt' in l for l in selected[control.name])>=6 and sum('restored context checkpoint' in l for l in selected[control.name])>=6
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
raw={str(p.relative_to(R)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['raw-sha256.json','verification.json']}
save(R/'raw-sha256.json',raw)
verification={'head':head,'source_unchanged':True,'stage28_binary_hashes_verified':32,'cpu_cases':len(cpu['cases']),'server_processes':len(s['server']),'server_responses':len(raw_responses),'loaded_libraries_verified':True,'model_stats_verified':True,'gpu_idle_after':True,'raw_index_sha256':sha(R/'raw-sha256.json')};save(R/'verification.json',verification)
out=DOC/'stage29';assert not out.exists();out.mkdir()
for name in ['README.md','run-server.py','run-ram.py','run-ram-controls.py','run-cpu.py','restore-replay.cpp','analyze-results.py','verify-archive.py','cpu-manifest.json','summary.json','verification.json','raw-sha256.json','gpu-after.csv']:shutil.copy2(R/name,out/name)
save(out/'requests.json',raw_requests);save(out/'responses-compact.json',raw_responses);save(out/'server-manifests.json',manifests);save(out/'selected-server-events.json',selected)
for label in cpu['cases']:
 dst=out/'cpu'/label;dst.mkdir(parents=True);shutil.copy2(R/'cpu'/label/'rows.jsonl',dst/'rows.jsonl')
index={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};save(out/'artifact-sha256.json',index)
shutil.copy2(R.parent/'STAGE29_REVIEW.md',DOC/'STAGE29_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/','](').replace('](llama_glm53_reasoning_budget/REVIEW.md)','](../glm53-reasoning-budget-20261008/REVIEW.md)'))
p=DOC/'README.md';t=p.read_text();t+='\n\n[Stage 29 report](STAGE29_REVIEW.md) records the user-requested GLM5NEXT cache reproducibility checks after commit 5a4de37b8. Production sources and installed binaries are unchanged. Results and scope limitations are in the report; stop for review before committing the test archive.\n';p.write_text(t)
print(json.dumps(verification,indent=2))
