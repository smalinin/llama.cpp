import hashlib,json,shutil
from pathlib import Path

R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007')
OUT=DOC/'stage17';assert not OUT.exists();OUT.mkdir()
files=[p for p in R.iterdir() if p.suffix in ['.py','.h','.patch'] or p.name in ['replay-control.cpp','explain-replay.cpp','explain-native.i32','README.md','.gitattributes','build.txt','explain-build.txt']]
files += [p for p in R.glob('*.json') if p.name!='raw-file-sha256.json']
files += list((R/'replay-output').glob('*.json*'))
files += list((R/'explain-replay-output').glob('*.json*'))
for p in files:
    d=OUT/p.relative_to(R);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
for config in ['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']:
    src=R/'free-runs'/config;dst=OUT/'free-runs'/config;dst.mkdir(parents=True)
    for p in src.glob('*.json'):
        value=json.loads(p.read_text())
        if p.name.endswith('-response.json') and 'completion_probabilities' in value:
            value.pop('completion_probabilities');value['completion_probabilities_omitted_from_review_archive']=True
        (dst/p.name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    for p in src.glob('*-prompt.txt'):shutil.copy2(p,dst/p.name)
    lines=(src/'server.log').read_text(errors='replace').splitlines()
    if not config.startswith('snapshot-'):
        (dst/'callback-events.txt').write_text('\n'.join(l for l in lines if l.startswith('DS14_EVENT '))+'\n')
    (dst/'server-completion.txt').write_text('\n'.join(l for l in lines if 'print_timing:' in l or 'cleaning up before exit' in l)+'\n')
lines=(R/'replay.log').read_text(errors='replace').splitlines()
(OUT/'replay-completion.txt').write_text('\n'.join(l for l in lines if l.startswith(('DONE ','REPLACED ','REPLACED_UPGATE ')))+'\n')
lines=(R/'explain-replay.log').read_text(errors='replace').splitlines()
(OUT/'explain-replay-completion.txt').write_text('\n'.join(l for l in lines if l.startswith(('DONE ','REPLACED ','REPLACED_UPGATE ')))+'\n')
shutil.copy2(R.parent/'STAGE17_REVIEW.md',DOC/'STAGE17_REVIEW.md')
(DOC/'PLAN.md').write_text((R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/',']('))
shutil.copy2(R/'repo-README.md',DOC/'README.md')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
