from array import array
import ctypes,hashlib,importlib.util,json,math
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'down-candidate-output';MODE='decode-scalar-fa-upgate-hc-router-down';V=129280
spec=importlib.util.spec_from_file_location('chain',R/'analyze-chain.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def load(p):
 v=array('f');v.frombytes(p.read_bytes());return v
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=R.parent/'stage14/extended-replay-output'/f'{a.MODE}-w1-logits.f32'
base=load(ref);assert len(base)==95*V
native=json.loads((R.parent/'stage14/free-runs/snapshot-off/baseline-greedy-1-response.json').read_text())['tokens']
summary={'scalar_control':{},'variants':{},'early_boundaries':{},'scope':'Teacher-forced95 rows, no draft/rollback; original scalar must retain Stage14 SHA. Diagnostic callback recomputes routed down/weight/sum after wide execution; timings are not production estimates.'}
summary['scalar_control']={'reference_sha256':sha(ref),'sha256':sha(D/f'{MODE}-w1-logits.f32')}
assert summary['scalar_control']['sha256']==summary['scalar_control']['reference_sha256']
for w in [1,2,4]:
 p=D/f'{MODE}-w{w}-logits.f32';values=load(p)
 rows=[json.loads(l) for l in (D/f'{MODE}-w{w}-rows.jsonl').read_text().splitlines()];assert len(rows)==95 and len(values)==len(base)
 differences=[i for i,r in enumerate(rows) if r['argmax']!=native[i]]
 result=a.compare(base,values);result.update(sha256=sha(p),argmax_difference_indices=differences,
     gap_These_minus_Wait_at87=values[87*V+10137]-values[87*V+39059],argmax_at87=rows[87]['argmax'])
 summary['variants'][str(w)]=result
 for target in [0,86]:
  for owner in ['ffn_moe_out-0','ffn_out-0']:
   raw=load(D/f'{MODE}-w{w}-input{target}-{owner}.f32')
   col=2 if target==86 and w==4 else 0
   refb=load(D/f'{MODE}-w1-input{target}-{owner}.f32')
   summary['early_boundaries'][f'input{target}-{owner}-w{w}']=a.compare(refb,raw[col*5120:(col+1)*5120])
summary['width2_vs4']=a.compare(load(D/f'{MODE}-w2-logits.f32'),load(D/f'{MODE}-w4-logits.f32'))
for w in [2,4]:
 counts=json.loads((D/f'{MODE}-w{w}-matmul-counts.json').read_text())
 routed={k:v for k,v in counts.items() if k.startswith('scalar-down-')}
 assert len(routed)==40 and all(v==3760//40 for v in routed.values()),routed
summary['down_recompute_counts']={'w1':0,'w2':3760,'w4':3760}
(R/'candidate-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
