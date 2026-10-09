#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import shutil
R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007')
OUT=DOC/'stage22'
assert not OUT.exists();OUT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
skip={'repo-README.md','model-partial-summary.json','free-partial-summary.json'}
for p in R.iterdir():
 if p.is_file() and p.name not in skip and (p.suffix in ['.py','.h','.patch','.json','.csv'] or p.name in ['README.md','.gitattributes','cache-replay.cpp','build.txt','replay-build.txt','cases.txt','cases-long.txt','cases-all.txt']):
  shutil.copy2(p,OUT/p.name)
shutil.copytree(R/'inputs',OUT/'inputs')
for root in ['model-output']:
 for p in (R/root).rglob('*'):
  if p.is_file() and (p.suffix in ['.json','.jsonl'] or p.name=='loaded-libraries.txt'):
   dst=OUT/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for config in ['integration-off','candidate-off','candidate-n1','candidate-n3']:
 src=R/'free-runs'/config;dst=OUT/'free-runs'/config;dst.mkdir(parents=True)
 for p in src.glob('*.json'):
  v=json.loads(p.read_text())
  if isinstance(v,dict) and 'completion_probabilities' in v:
   probabilities=v.pop('completion_probabilities')
   v['completion_probabilities_omitted_from_review_archive']=True
   v['completion_probabilities_sha256']=hashlib.sha256(json.dumps(probabilities,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
  (dst/p.name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 for p in src.glob('*-prompt.txt'):shutil.copy2(p,dst/p.name)
 lines=(src/'server.log').read_text(errors='replace').splitlines()
 (dst/'callback-events.txt').write_text('\n'.join(l.rstrip() for l in lines if l.startswith('DS14_EVENT '))+'\n')
 (dst/'server-cache-events.txt').write_text('\n'.join(l.rstrip() for l in lines if any(k in l for k in ['checkpoint','cache','print_timing:','cleaning up before exit']))+'\n')
for name in ['replay-run.log','free-run.log']:
 lines=(R/name).read_text(errors='replace').splitlines()
 (OUT/(name+'.completion.txt')).write_text('\n'.join(l.rstrip() for l in lines if l.startswith(('START ','READY ','DONE ','FINISH ')))+'\n')
for name in ['replay.log']:
 lines=(R/name).read_text(errors='replace').splitlines()
 (OUT/(name+'.completion.txt')).write_text('\n'.join(l.rstrip() for l in lines if l.startswith(('DONE ','FAIL ')))+'\n')
failed=OUT/'failed-attempt-1';failed.mkdir()
for name in ['failure.json','raw-layout.h','replay-build-manifest.json','replay-run-manifest.json']:
 shutil.copy2(R/'failed-attempt-1'/name,failed/name)
untrimmed=OUT/'untrimmed-restore-attempt';untrimmed.mkdir()
for name in ['failure.json','cache-replay.cpp','replay-build-manifest.json','replay-run-manifest.json','replay-long-run-manifest.json']:
 shutil.copy2(R/'untrimmed-restore-attempt'/name,untrimmed/name)
shutil.copy2(R.parent/'STAGE22_REVIEW.md',DOC/'STAGE22_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
