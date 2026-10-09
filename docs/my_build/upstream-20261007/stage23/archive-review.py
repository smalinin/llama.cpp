#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007');OUT=DOC/'stage23'
assert not OUT.exists();OUT.mkdir()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p in R.iterdir():
 if p.is_file() and p.name!='repo-README.md' and (p.suffix in ['.py','.h','.cpp','.json','.csv'] or p.name in ['README.md','.gitattributes','cases.txt','focused-cases.txt']):shutil.copy2(p,OUT/p.name)
for root in ['model-output','focused-output']:
 for p in (R/root).rglob('*'):
  if p.is_file() and (p.suffix in ['.json','.jsonl'] or p.name=='loaded-libraries.txt'):
   dst=OUT/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for root in ['projection-inputs','projection-output','indexer-output','attention-output']:
 for p in (R/root).rglob('*'):
  if p.is_file() and (p.suffix in ['.json','.jsonl'] or p.name=='loaded-libraries.txt'):
   dst=OUT/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for name in ['replay.log','focused.log']:
 lines=(R/name).read_text(errors='replace').splitlines()
 (OUT/(name+'.completion.txt')).write_text('\n'.join(l.rstrip() for l in lines if l.startswith(('DONE ','FAIL ')))+'\n')
shutil.copy2(R.parent/'STAGE23_REVIEW.md',DOC/'STAGE23_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
