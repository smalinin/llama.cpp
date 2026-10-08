#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review-artifacts';OUT.mkdir(exist_ok=True)
files=['README.md','ffn-controls.cpp','ffn-profile.cpp','target-ffn-controls.cpp',
       'run-controls.py','run-target-ffn.py','analyze-controls.py','analyze-target.py','analyze-profile.py','verify-integrity.py','curate-artifacts.py',
       'build-commands.json','controls-summary.json','kernel-summary.json','integrity.json','raw-artifact-sha256.json',
       'target-runtime-summary.json','target-events.txt','target-ffn-manifest.json','target-ffn-output/summary.json','target-ffn-output/loaded-libraries.txt']
for label in ['controls-ada','controls-ampere','controls-ada-profile']:
 files.extend([label+'-manifest.json',label+'/results.jsonl',label+'/timings.jsonl'])
for f in sorted((ROOT/'target-ffn-output').glob('*-rows.jsonl')):files.append(str(f.relative_to(ROOT)))
for name in files:
 dst=OUT/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
for label in ['controls-ada','controls-ampere','controls-ada-profile']:
 name=label+'.txt';shutil.copy2(ROOT/(label+'.log'),OUT/name);files.append(name)
hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in sorted(files)}
(OUT/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'files':len(files)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
