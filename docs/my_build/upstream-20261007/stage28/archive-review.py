#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007');OUT=DOC/'stage28'
assert json.loads((R/'integrity.json').read_text())['completed_server_control_requests']==48
assert not OUT.exists();OUT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
skip={'repo-README.md','partial-summary.json'}
for p in R.iterdir():
 if p.is_file() and p.name not in skip and (p.suffix in ['.py','.cpp','.patch','.json','.csv'] or p.name in ['README.md','.gitattributes']):shutil.copy2(p,OUT/p.name)
for name in ['after-unified','reference-unified','before-separated','after-separated','final-pool-unified','final-pool-reference','final-pool-separated','mtp-final-unified','mtp-reference-unified','mtp-before-separated','mtp-final-separated','mtp-before-single','mtp-final-single']:shutil.copy2(R/(name+'.jsonl'),OUT/(name+'.jsonl'))
for label in ['before-unified','after-unified','reference-unified','after-separated','mtp-unified','mtp-final-unified']:
 src=R/'server-runs'/label;dst=OUT/'server-runs'/label;dst.mkdir(parents=True)
 shutil.copy2(src/'control-manifest.json',dst/'control-manifest.json')
 src=src/'glm5next-spec-1';dst=dst/'glm5next-spec-1';dst.mkdir()
 for p in src.glob('*.json'):shutil.copy2(p,dst/p.name)
 lines=(src/'server.log').read_text(errors='replace').splitlines()
 (dst/'selected-server-events.txt').write_text('\n'.join(l.rstrip() for l in lines if any(k in l for k in ['incremental pool-key','GGML_ASSERT','print_timing:','checkpoint','cache reuse','cleaning up before exit']))+'\n')
for name in ['server-run.log','final-server-run.log','pool-run.log','final-pool-run.log','mtp-run.log','existing-run.log','final-existing-run.log']:shutil.copy2(R/name,OUT/(name+'.txt'))
shutil.copy2(R.parent/'STAGE28_REVIEW.md',DOC/'STAGE28_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/','](').replace('](llama_glm53_reasoning_budget/REVIEW.md)','](../glm53-reasoning-budget-20261008/REVIEW.md)'))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()};(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
