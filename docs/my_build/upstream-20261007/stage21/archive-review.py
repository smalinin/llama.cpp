#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import shutil
R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007');OUT=DOC/'stage21'
assert not OUT.exists();OUT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in R.iterdir() if p.is_file() and (p.suffix in ['.py','.h','.patch','.json','.csv'] or p.name in ['README.md','.gitattributes','build.txt','build-extent.txt']) and p.name not in ['repo-README.md','free-partial-summary.json']]
for p in files:shutil.copy2(p,OUT/p.name)
for config in ['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n1-remaining','candidate-n3']:
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
 if config!='snapshot-off':(dst/'callback-events.txt').write_text('\n'.join(l.rstrip() for l in lines if l.startswith('DS14_EVENT '))+'\n')
 (dst/'cache-failure.txt').write_text('\n'.join(l.rstrip() for l in lines if any(k in l for k in ['DS21_EXTENT ', 'unsupported cache extent','compute buffer size']))+'\n')
 (dst/'server-cache-events.txt').write_text('\n'.join(l.rstrip() for l in lines if any(k in l for k in ['checkpoint','cache','print_timing:','cleaning up before exit']))+'\n')
for name in ['free-run.log','followup-run.log']:
 lines=(R/name).read_text().splitlines()
 (OUT/(name+'.completion.txt')).write_text('\n'.join(l.rstrip() for l in lines if l.startswith(('START ','READY ','DONE ','FINISH ')))+'\n')
shutil.copy2(R.parent/'STAGE21_REVIEW.md',DOC/'STAGE21_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
