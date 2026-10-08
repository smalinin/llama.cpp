#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
D=Path(sys.argv[1])
metric=ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
V=129280

def read(p):
    x=array('f')
    with p.open('rb') as f: x.fromfile(f,p.stat().st_size//4)
    return x

def compare(a,b):
    assert len(a)==len(b)
    o=(ctypes.c_double*5)()
    metric(a.buffer_info()[0],b.buffer_info()[0],len(a),0,o)
    return {'max_abs':o[2],'rms':math.sqrt(o[0]/len(a)),
            'reference_rms':math.sqrt(o[1]/len(a)),'nonfinite':int(o[4]),
            'bit_identical':a.tobytes()==b.tobytes()}

summary={'forced_prefix':'Stage9 current native HTTP output','rows':65,'variants':{}}
for p in sorted(D.glob('*-rows.jsonl')):
    label=p.name.removesuffix('-rows.jsonl')
    mode=label.split('-w')[0].removesuffix('-features')
    base_label=mode+'-w1'
    if not (D/(base_label+'-rows.jsonl')).exists(): continue
    arows=[json.loads(l) for l in (D/(base_label+'-rows.jsonl')).read_text().splitlines()]
    brows=[json.loads(l) for l in p.read_text().splitlines()]
    a=read(D/(base_label+'-logits.f32'));b=read(D/(label+'-logits.f32'))
    result=compare(a,b)
    result.update(argmax_difference_indices=[i for i,(x,y) in enumerate(zip(arows,brows)) if x['argmax']!=y['argmax']],
                  width1_difference_vs_forced=[i for i,x in enumerate(arows) if x['argmax']!=x['reference']],
                  sha256=hashlib.sha256(b).hexdigest(),index19=brows[19],
                  index19_verify_minus_check=b[19*V+23393]-b[19*V+4085],
                  first_wide_row17=compare(a[17*V:18*V],b[17*V:18*V]),
                  row19=compare(a[19*V:20*V],b[19*V:20*V]))
    summary['variants'][label]=result

for w in [1,2,4]:
    p=D/f'default-features-w{w}-features.f32'
    if not p.exists(): continue
    if w==1: features_base=read(p)
    else:
        x=read(p)
        assert len(x)==len(features_base)==65*41*5120
        per=[]
        for row in [1,17,19,33]:
            for layer in range(41):
                start=(row*41+layer)*5120
                m=compare(features_base[start:start+5120],x[start:start+5120])
                m.update(row=row,layer=layer)
                per.append(m)
        summary.setdefault('layer_features',{})[f'width{w}']=per
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k: {a:b for a,b in v.items() if a not in ('sha256',)} for k,v in summary['variants'].items()},indent=2))
