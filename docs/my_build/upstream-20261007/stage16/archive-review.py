from pathlib import Path
import hashlib,json,shutil

R=Path(__file__).resolve().parent
DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007')
OUT=DOC/'stage16'
assert not OUT.exists()
OUT.mkdir()
files=[p for p in R.iterdir() if p.suffix in ['.cpp','.h','.py','.txt'] or p.name=='README.md']
files += [p for p in R.glob('*.json') if p.name!='raw-file-sha256.json']
for directory in ['chain-capture-output','attention-capture-output','compressor-candidate-output']:
    files += [p for p in (R/directory).rglob('*') if p.suffix in ['.json','.jsonl']]
for directory in ['down-cost-ada','down-cost-ampere','compressor-kv-ada','compressor-kv-ampere','compressor-gate-ada','compressor-gate-ampere','attention-w1-ada','attention-w2-ada','attention-w4-ada','attention-w2-swap-ada','attention-w1-ampere','attention-w2-ampere','attention-w4-ampere','attention-w2-swap-ampere']:
    files += [p for p in (R/directory).iterdir() if p.suffix in ['.jsonl','.txt']]
for directory in ['compressor-inputs','attention-inputs']:
    for p in (R/directory).rglob('*'):
        if p.suffix in ['.json','.txt'] or (directory=='compressor-inputs' and p.suffix=='.f32'):files.append(p)
for p in files:
    dst=OUT/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
shutil.copy2(R.parent/'STAGE16_REVIEW.md',DOC/'STAGE16_REVIEW.md')
plan=(R.parent.parent/'LLAMA_CPP_UPSTREAM_PLAN.md').read_text().replace('](llama_upstream_review/','](')
(DOC/'PLAN.md').write_text(plan)
readme=(DOC/'README.md').read_text()
readme=readme.replace('[Stage 15 report](STAGE15_REVIEW.md) was approved by the user for commit.', '[Stage 15 report](STAGE15_REVIEW.md) was approved by the user and committed as d0060fd95.')
readme+='''

[Stage 16 report](STAGE16_REVIEW.md) is prepared for review before commit. Adding scalar compressor KV/gate to the Stage15 diagnostic combination makes all95 full logits at widths1/2/4 bit-identical to native. Fixed-input BF16 projections and a one-row K/V replacement reproduce the remaining attention difference. All15 model replays, 11 reference SHA controls, 140 isolated cases and32 snapshot hashes pass. Direct scalar down graphs cost1.56-1.82 times native locally, excluding callback duplication. Free DSpark generation, width3 and production throughput remain unverified. No production source, installed server or profile change.

Stage15 is committed; Stage16 awaits review before commit. After review, check native-off/replay integration and free DSpark N=1/N=3 on four prompts before evaluating a production implementation and its throughput.
'''
(DOC/'README.md').write_text(readme)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=OUT/'attention-fragment.cpp'
raw_sha=sha(p)
p.write_text(p.read_text().rstrip()+'\n')
(OUT/'archive-normalization.json').write_text(json.dumps({'attention-fragment.cpp':{'operation':'Trim trailing blank line in the standalone archived fragment. Full compiled source and original workspace fragment are unchanged.','raw_sha256':raw_sha,'archived_sha256':sha(p)}},indent=2)+'\n')
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
assert all(sha(OUT/n)==h for n,h in index.items())
print(json.dumps({'archived_files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'review':str(DOC/'STAGE16_REVIEW.md')}))
