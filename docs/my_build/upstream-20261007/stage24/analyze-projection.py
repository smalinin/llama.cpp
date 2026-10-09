#!/usr/bin/env python3
from pathlib import Path
import hashlib,json

R=Path(__file__).resolve().parent
OLD=R.parent/'stage23'
D=R/'projection-output'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

manifest=json.loads((OLD/'projection-inputs/manifest.json').read_text())
assert all(sha(OLD/'projection-inputs'/name)==v['sha256'] for name,v in manifest['files'].items())
rows=[json.loads(line) for line in (D/'results.jsonl').read_text().splitlines()]
assert len(rows)==11 and len({r['label'] for r in rows})==11
assert all(r['repeat_stable'] and r['max_vs_scalar']==0 and r['column_max']==0 for r in rows)
size=manifest['shape'][1]*4
for source in [1,2,4]:
    scalar=(D/f'source{source}-w1-repeat.f32').read_bytes()
    assert len(scalar)==size
    for width in [1,2,4]:
        assert (D/f'source{source}-w{width}-repeat.f32').read_bytes()==scalar*width
reference=(OLD/'projection-output/source1-w1-repeat.f32').read_bytes()
assert (D/'source1-w1-repeat.f32').read_bytes()==reference
assert (D/'source2-w2-actual.f32').read_bytes()==reference*2
actual=(D/'source4-w4-actual.f32').read_bytes()
column=manifest['wide_column']
assert len(actual)==size*4 and actual[column*size:(column+1)*size]==reference
out={'variants':11,'repeats_stable':11,'repeat_columns_exact':True,'scalar_sha_preserved':True,
     'wide_actual_target_column_exact':True,'header_sha256':sha(R/'general-callback.h')}
(R/'projection-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
