#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review-artifacts'
OUT.mkdir(exist_ok=True)
files=['README.md','prepare-ffn.py','run-ffn.py','run-replay.py','run-cpu-delta.py','analyze-ffn.py','verify-integrity.py','curate-artifacts.py',
       'ffn-replay.cpp','down-replay.cpp','q8-capture.cpp','cpu-delta.cpp','ffn-bench.cpp',
       'build-commands.json','excluded-attempts.json','ffn-summary.json','benchmark-summary.json','kernel-summary.json',
       'integrity.json','raw-artifact-sha256.json','cpu-delta-manifest.json','inputs/manifest.json','inputs/weight-index.json','inputs/weight-index.tsv']
for label in ['ffn-ada','ffn-ampere','ffn-ada-no-fusion','ffn-ada-profile','down-ada','down-ampere','q8-ada','q8-ampere','bench-ada','bench-ampere']:
    files.extend([label+'-manifest.json',label+'.log'])
    if (ROOT/label/'results.jsonl').exists():files.append(label+'/results.jsonl')
    if (ROOT/label/'timings.jsonl').exists():files.append(label+'/timings.jsonl')
for name in files:
    dst=OUT/name;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/name,dst)
hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in sorted(files)}
(OUT/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'files':len(files)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
