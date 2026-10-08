from array import array
import hashlib,json,runpy
from pathlib import Path

R=Path(__file__).resolve().parent;D=R/'explain-replay-output';V=129280
BASE='decode-scalar-fa-upgate-hc-router-down-'
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
def load(p):
    a=array('f');a.frombytes(p.read_bytes());return a
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reference=D/f'{BASE}compressor-w1-logits.f32';base=load(reference);assert len(base)==256*V
native=json.loads((R/'free-runs/snapshot-off/explain-greedy-2-response.json').read_text())['tokens']
result={'scope':'Native256 explain teacher-forced history, fixed widths, no draft or rollback. Scalar argmax must reproduce all256 native output IDs.','controls':{},'variants':{}}
for mode in ['compressor','float2d']:
    for w in [1,2,3,4]:
        label=f'{BASE}{mode}-w{w}';p=D/f'{label}-logits.f32';a=load(p)
        rows=[json.loads(l) for l in (D/f'{label}-rows.jsonl').read_text().splitlines()]
        assert len(rows)==256 and len(a)==len(base)
        differences=[i for i,r in enumerate(rows) if r['argmax']!=native[i]]
        numeric=[i for i in range(256) if a[i*V:(i+1)*V].tobytes()!=base[i*V:(i+1)*V].tobytes()]
        v=compare(base,a);v.update(sha256=sha(p),argmax_difference_indices=differences,
            numeric_difference_indices=numeric,first_numeric=None if not numeric else dict(index=numeric[0],**compare(base[numeric[0]*V:(numeric[0]+1)*V],a[numeric[0]*V:(numeric[0]+1)*V])),
            argmax_at244=rows[244]['argmax'],gap_conditions_minus_operations_at244=a[244*V+4132]-a[244*V+7574])
        if w==1:
            assert not differences and v['bit_identical']
            result['controls'][mode]={'all256_native_argmax_exact':True,'scalar_sha256':sha(p),'reference_sha256':sha(reference)}
        result['variants'][label]=v
(R/'explain-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'controls':result['controls'],'variants':{k:{s:v[s] for s in ['max_abs','rms','bit_identical','argmax_difference_indices','first_numeric','argmax_at244','gap_conditions_minus_operations_at244']} for k,v in result['variants'].items()}},indent=2))
