#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil

R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007')
OUT=DOC/'stage27'
assert json.loads((R/'integrity.json').read_text())['long_n3_native_answers_exact']==2
assert not OUT.exists();OUT.mkdir()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
skip={'repo-README.md','replay-partial-summary.json','llama-kv-cache.cpp'}
for p in R.iterdir():
    if p.is_file() and p.name not in skip and (p.suffix in ['.py','.h','.cpp','.patch','.json','.csv'] or p.name in ['README.md','.gitattributes','cases.txt','verify-cases.txt']):shutil.copy2(p,OUT/p.name)
for directory in ['model-output']:
    src=R/directory;dst=OUT/directory;dst.mkdir()
    shutil.copy2(src/'loaded-libraries.txt',dst/'loaded-libraries.txt')
    for case in sorted(src.iterdir()):
        if not case.is_dir():continue
        target=dst/case.name;target.mkdir()
        for p in case.iterdir():
            if p.suffix in ['.json','.jsonl']:shutil.copy2(p,target/p.name)
src=R/'server-runs/candidate-n3';dst=OUT/'server-runs/candidate-n3';dst.mkdir(parents=True)
for p in src.glob('*.json'):
    value=json.loads(p.read_text())
    if isinstance(value,dict) and 'completion_probabilities' in value:
        probabilities=value.pop('completion_probabilities')
        value['completion_probabilities_omitted_from_review_archive']=True
        value['completion_probabilities_sha256']=hashlib.sha256(json.dumps(probabilities,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    (dst/p.name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(dst/'capture').mkdir();shutil.copy2(src/'capture/events.jsonl',dst/'capture/events.jsonl')
lines=(src/'server.log').read_text(errors='replace').splitlines()
(dst/'callback-events.txt').write_text('\n'.join(l.rstrip() for l in lines if l.startswith('DS14_EVENT '))+'\n')
(dst/'server-cache-events.txt').write_text('\n'.join(l.rstrip() for l in lines if any(k in l for k in ['checkpoint','cache','print_timing:','cleaning up before exit']))+'\n')
for name in ['replay.log','server-run.log']:
    lines=(R/name).read_text(errors='replace').splitlines()
    (OUT/(name+'.completion.txt')).write_text('\n'.join(l.rstrip() for l in lines if l.startswith(('START ','READY ','DONE ','FINISH ')))+'\n')
shutil.copy2(R.parent/'STAGE27_REVIEW.md',DOC/'STAGE27_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/','](').replace('](llama_glm53_reasoning_budget/REVIEW.md)','](../glm53-reasoning-budget-20261008/REVIEW.md)'))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
