from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007');OUT=DOC/'stage20'
assert not OUT.exists();OUT.mkdir()
files=[p for p in R.iterdir() if p.is_file() and (p.suffix in ['.py','.h','.cpp','.json','.csv'] or p.name=='README.md') and p.name!='repo-README.md']
files+=list(R.glob('*build*.txt'))
files+=[p for p in (R/'model-output').rglob('*') if p.is_file() and (p.suffix in ['.json','.jsonl'] or p.name=='loaded-libraries.txt')]
for p in set(files):
 d=OUT/p.relative_to(R);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
shutil.copy2(R.parent/'stage18/attention-replay.cpp',OUT/'reused-stage18-attention-replay.cpp')
for arch in ['ada0','ada2','ampere']:
 for prefix in ['attention-','attention-profile-']:
  d=OUT/f'{prefix}{arch}';d.mkdir();shutil.copy2(R/f'{prefix}{arch}/results.jsonl',d/'results.jsonl')
lines=(R/'replay.log').read_text(errors='replace').splitlines()
(OUT/'model-completion.txt').write_text('\n'.join(l for l in lines if l.startswith(('DONE ','REPLACED ','REPLACED_UPGATE ','PADDING_CROPS ','COMPRESSED_PADDING_CROPS ')))+'\n')
(OUT/'model-device-assignment.txt').write_text('\n'.join(l for l in lines if l.startswith('load_tensors: layer '))+'\n')
shutil.copy2(R.parent/'STAGE20_REVIEW.md',DOC/'STAGE20_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
