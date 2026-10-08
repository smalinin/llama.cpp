from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;DOC=Path('/home/sergei/Github/llama.cpp/docs/my_build/upstream-20261007');OUT=DOC/'stage15'
assert not OUT.exists();OUT.mkdir()
files=[p for p in R.iterdir() if p.suffix in ['.cpp','.h','.py'] or p.name=='README.md']
files += [R/n for n in ['build-manifest.json','candidate-build-manifest.json','components-build-manifest.json','build.log','candidate-build.log','components-build.log','chain-capture-manifest.json','down-candidate-manifest.json','rejected-build-manifest.json','rejected-chain-capture-manifest.json','rejected-capture-controls.json','chain-summary.json','chain-overview.json','expert-selection-summary.json','isolated-summary.json','kernel-summary.json','candidate-summary.json','isolated-manifest.json','isolated-profile-manifest.json','components-manifest.json','integrity.json']]
for name in ['manifest.json','weight-index.json','weight-index.tsv']:files.append(R/'inputs'/name)
for p in (R/'chain-capture-output').rglob('*'):
 if p.suffix in ['.json','.jsonl']:files.append(p)
for p in (R/'down-candidate-output').glob('*'):
 if p.suffix in ['.json','.jsonl']:files.append(p)
for label in ['down0-ada','down0-ampere','down0-ada-profile','down0-ampere-profile','down-ada','down-ampere','weighted-ada','weighted-ampere']:
 src=R/label/'results.jsonl';dst=OUT/src.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True)
 if label.startswith(('down-','weighted-')):
  rows=[json.loads(l) for l in src.read_text().splitlines()]
  for row in rows:row.pop('max_vs_capture',None);row['capture_comparison_performed']=False
  dst.write_text(''.join(json.dumps(row)+'\n' for row in rows))
 else:shutil.copy2(src,dst)
for p in files:
 dst=OUT/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
shutil.copy2(R.parent/'STAGE15_REVIEW.md',DOC/'STAGE15_REVIEW.md')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'artifact-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
assert all(sha(OUT/n)==h for n,h in index.items())
print(json.dumps({'archived_files':len(index)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'review':str(DOC/'STAGE15_REVIEW.md')}))
