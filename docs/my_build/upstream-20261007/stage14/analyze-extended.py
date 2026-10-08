#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;D=ROOT/'extended-replay-output';V=129280;MODE='decode-scalar-fa-upgate-hc-router'
metric=ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics;metric.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
def compare(a,b):
 assert len(a)==len(b);o=(ctypes.c_double*5)();metric(a.buffer_info()[0],b.buffer_info()[0],len(a),0,o);assert o[4]==0
 return {'max_abs':o[2],'rms':math.sqrt(o[0]/len(a)),'bit_identical':a.tobytes()==b.tobytes()}
base=read(D/f'{MODE}-w1-logits.f32');assert len(base)==95*V
native=json.loads((ROOT/'free-runs/snapshot-off/baseline-greedy-1-response.json').read_text());assert len(native['tokens'])==95
out={'rows':95,'controls':{},'variants':{},'width2_vs4':{},'scope':'Teacher-forced full native answer, no draft or rollback. First65 rows must retain Stage13 SHA.'}
rows_by_width={}
for w in [1,2,4]:
 label=f'{MODE}-w{w}';p=D/f'{label}-logits.f32';a=read(p);assert len(a)==95*V
 rows=[json.loads(l) for l in (D/f'{label}-rows.jsonl').read_text().splitlines()];assert len(rows)==95;rows_by_width[w]=rows
 prefix=hashlib.sha256(a[:65*V].tobytes()).hexdigest();old=sha(ROOT.parent/'stage13/target-matmul-output'/f'{label}-logits.f32');assert prefix==old
 out['controls'][str(w)]={'first65_sha256':prefix,'stage13_sha256':old,'bit_identical':True}
 differences=[i for i,r in enumerate(rows) if r['argmax']!=native['tokens'][i]]
 result=compare(base,a);result.update(sha256=sha(p),argmax_difference_indices=differences,first_difference=next(iter(differences),None),different_rows=[{'index':i,'scalar_argmax':native['tokens'][i],'wide_argmax':rows[i]['argmax'],'wide_second':rows[i]['second'],'wide_gap':rows[i]['gap'],'scalar_logit_difference_reference_minus_wide':base[i*V+native['tokens'][i]]-base[i*V+rows[i]['argmax']],'wide_logit_difference_reference_minus_wide':a[i*V+native['tokens'][i]]-a[i*V+rows[i]['argmax']]} for i in differences])
 if w==1:assert not differences
 out['variants'][str(w)]=result
out['width2_vs4']=compare(read(D/f'{MODE}-w2-logits.f32'),read(D/f'{MODE}-w4-logits.f32'))
(ROOT/'extended-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
