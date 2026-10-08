from array import array
import hashlib,json,runpy
from pathlib import Path

R=Path(__file__).resolve().parent
helpers=runpy.run_path(str(R/'analyze-chain.py'));compare=helpers['compare']
MODE='decode-scalar-fa-upgate-hc-router-down';V=129280
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    a=array('f');a.frombytes(p.read_bytes());return a
base=load(R.parent/'stage15/down-candidate-output'/f'{MODE}-w1-logits.f32')
native=json.loads((R.parent/'stage14/free-runs/snapshot-off/baseline-greedy-1-response.json').read_text())['tokens']
assert len(base)==95*V and len(native)==95
numeric={}
for w in [2,4]:
    a=load(R/'chain-capture-output'/f'{MODE}-w{w}-logits.f32')
    diffs=[dict(index=i,**compare(base[i*V:(i+1)*V],a[i*V:(i+1)*V])) for i in range(95) if base[i*V:(i+1)*V].tobytes()!=a[i*V:(i+1)*V].tobytes()]
    numeric[str(w)]={'different_indices':[r['index'] for r in diffs],'first_differences':diffs[:6]}
(R/'numeric-row-summary.json').write_text(json.dumps(numeric,indent=2)+'\n')

D=R/'compressor-candidate-output'
result={'scope':'Teacher-forced95 rows, unchanged native history; no draft, rollback, HTTP performance or free-generation claim.','controls':{},'variants':{}}
for suffix in ['', '-compressor', '-float2d']:
    values={}
    for w in [1,2,4]:
        label=f'{MODE}{suffix}-w{w}'
        p=D/f'{label}-logits.f32';a=load(p);values[w]=a
        assert len(a)==len(base)
        if suffix=='' or w==1:
            ref=R.parent/'stage15/down-candidate-output'/f'{MODE}-w{w if suffix=="" else 1}-logits.f32'
            assert sha(p)==sha(ref)
            result['controls'][label]={'sha256':sha(p),'reference_sha256':sha(ref),'bit_identical':True}
        rows=[json.loads(l) for l in (D/f'{label}-rows.jsonl').read_text().splitlines()]
        assert len(rows)==95
        numeric=[i for i in range(95) if a[i*V:(i+1)*V].tobytes()!=base[i*V:(i+1)*V].tobytes()]
        argmax=[i for i,r in enumerate(rows) if r['argmax']!=native[i]]
        result['variants'][label]=dict(compare(base,a),sha256=sha(p),numeric_difference_indices=numeric,argmax_difference_indices=argmax,
            first_numeric=None if not numeric else dict(index=numeric[0],**compare(base[numeric[0]*V:(numeric[0]+1)*V],a[numeric[0]*V:(numeric[0]+1)*V])))
    result.setdefault('width2_vs4',{})[suffix or 'base']=compare(values[2],values[4])
(R/'candidate-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'controls':len(result['controls']),'variants':{k:{s:v[s] for s in ['max_abs','rms','bit_identical','first_numeric','argmax_difference_indices']} for k,v in result['variants'].items()},'width2_vs4':result['width2_vs4']},indent=2))
