#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil

R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007')
OUT=DOC/'stage25'
assert not OUT.exists();OUT.mkdir()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

skip={'repo-README.md','free-partial-summary.json'}
for p in R.iterdir():
    if p.is_file() and p.name not in skip and (p.suffix in ['.py','.h','.cpp','.patch','.json','.csv'] or p.name in ['README.md','.gitattributes','build.txt']):
        shutil.copy2(p,OUT/p.name)
for config in ['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']:
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
lines=(R/'free-run.log').read_text(errors='replace').splitlines()
(OUT/'free-run.log.completion.txt').write_text('\n'.join(l.rstrip() for l in lines if l.startswith(('START ','READY ','DONE ','FINISH ')))+'\n')
shutil.copy2(R.parent/'STAGE25_REVIEW.md',DOC/'STAGE25_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
