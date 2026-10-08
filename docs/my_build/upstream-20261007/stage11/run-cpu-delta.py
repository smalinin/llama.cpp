#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
w=next(x for x in json.loads((ROOT/'inputs/weight-index.json').read_text()) if x['tensor']=='blk.2.ffn_down_exps.weight')
size=w['bytes']//384
offset=w['offset']+217*size
with open(w['source'],'rb') as f:
    f.seek(offset)
    data=f.read(size)
assert len(data)==size
weight=ROOT/'inputs/down-expert217.bin'
weight.write_bytes(data)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'original':w,'expert_id':217,'slot':2,'expert_offset':offset,'bytes':size,'sha256':sha(weight),
   'source_sha256':sha(ROOT/'cpu-delta.cpp'),'binary_sha256':sha(ROOT/'cpu-delta'),'commands':{}}
for arch in ['ada','ampere']:
    command=list(map(str,[ROOT/'cpu-delta',weight,ROOT/('ffn-'+arch),ROOT/('q8-'+arch),ROOT/('cpu-delta-'+arch)]))
    subprocess.run(command,check=True)
    m['commands'][arch]=command
(ROOT/'cpu-delta-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
