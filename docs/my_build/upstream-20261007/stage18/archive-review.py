from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007');OUT=DOC/'stage18'
assert not OUT.exists();OUT.mkdir()
files=[p for p in R.iterdir() if p.is_file() and (p.suffix in ['.py','.h','.cpp','.patch','.json'] or p.name in ['README.md','.gitattributes']) and p.name not in ['raw-file-sha256.json','repo-README.md','server-context.cpp','server-context-v2.cpp']]
files += [p for p in R.glob('*build*.txt')]
for directory in ['capture-output','candidate-output','candidate-capture-output','v2-output']:
 files += [p for p in (R/directory).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl']]
for p in set(files):
 d=OUT/p.relative_to(R);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
for config in ['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']:
 src=R/'free-runs'/config;dst=OUT/'free-runs'/config;dst.mkdir(parents=True)
 for p in src.glob('*.json'):
  v=json.loads(p.read_text())
  if p.name.endswith('-response.json') and 'completion_probabilities' in v:
   v.pop('completion_probabilities');v['completion_probabilities_omitted_from_review_archive']=True
  (dst/p.name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 for p in src.glob('*-prompt.txt'):shutil.copy2(p,dst/p.name)
 lines=(src/'server.log').read_text(errors='replace').splitlines()
 if not config.startswith('snapshot-'):(dst/'callback-events.txt').write_text('\n'.join(l for l in lines if l.startswith('DS14_EVENT '))+'\n')
 (dst/'server-completion.txt').write_text('\n'.join(l for l in lines if 'print_timing:' in l or 'cleaning up before exit' in l)+'\n')
for name in ['attention-ada','attention-ampere','attention-profile-ada','attention-profile-ampere','layer20-ada','layer20-ampere','layer20-attention-profile-ada','layer20-attention-profile-ampere']:
 d=OUT/name;d.mkdir();shutil.copy2(R/name/'results.jsonl',d/'results.jsonl')
for name in ['capture','candidate','candidate-capture','v2']:
 lines=(R/f'{name}.log').read_text(errors='replace').splitlines()
 (OUT/f'{name}-completion.txt').write_text('\n'.join(l for l in lines if l.startswith(('DONE ','REPLACED ','REPLACED_UPGATE ','PADDING_CROPS ','COMPRESSED_PADDING_CROPS ')))+'\n')
shutil.copy2(R.parent/'STAGE18_REVIEW.md',DOC/'STAGE18_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
