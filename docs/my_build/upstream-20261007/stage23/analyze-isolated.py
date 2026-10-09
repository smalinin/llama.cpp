#!/usr/bin/env python3
from pathlib import Path
from array import array
import hashlib,json,math,runpy,struct
R=Path(__file__).resolve().parent
h=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'));compare=h['compare']
def read(p):a=array('f');a.frombytes(p.read_bytes());return a
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
result={'projection':{},'indexer':{},'attention':{}}
for w,label in [(1,'source1-w1-repeat'),(4,'source4-w4-actual')]:
 values=read(R/'projection-output'/f'{label}.f32');scaled=array('f',(v/64 for v in values));cap=read(R/'projection-inputs'/f'captured-w{w}.f32')
 result['projection'][f'capture_w{w}']=compare(scaled,cap)
a=read(R/'projection-output/source1-w1-repeat.f32');b=read(R/'projection-output/source1-w4-repeat.f32')[:32]
result['projection']['frozen_width']=compare(a,b)
result['projection']['scaled_width']=compare(array('f',(v/64 for v in a)),array('f',(v/64 for v in b)))
result['projection']['repeat_stable']=all(v['repeat_stable'] for v in map(json.loads,(R/'projection-output/results.jsonl').read_text().splitlines()))
result['projection']['variants']=len((R/'projection-output/results.jsonl').read_text().splitlines())
assert result['projection']['capture_w1']['bit_identical'] and result['projection']['capture_w4']['bit_identical']
assert not result['projection']['frozen_width']['bit_identical']
for kind,reference in [('scalar','scalar'),('wide','wide'),('corrected','scalar'),('reverse','wide')]:
 for width in [1,4]:
  label=f'{kind}-w{width}';checks={}
  for suffix in ['scores.f32','top.i32']:
   actual=(R/'indexer-output'/f'{label}-{suffix}').read_bytes();expected=(R/'indexer-output'/f'{reference}-w1-{suffix}').read_bytes()*width
   checks[suffix]=actual==expected
  checks['last_selected']=struct.unpack('<2048i',(R/'indexer-output'/f'{label}-top.i32').read_bytes()[:8192])[-1]
  if kind in ['scalar','wide']:
   for suffix,name in [('scores.f32','captured-scores.f32'),('top.i32','captured-top.i32')]:checks['captured_'+suffix]=(R/'indexer-output'/f'{label}-{suffix}').read_bytes()==(R/'indexer-inputs'/label/name).read_bytes()*width
  result['indexer'][label]=checks
  assert all(v for k,v in checks.items() if k!='last_selected')
scalar=read(R/'indexer-output/scalar-w1-scores.f32');wide=read(R/'indexer-output/wide-w1-scores.f32')
finite=[i for i,v in enumerate(scalar) if math.isfinite(v)];assert finite==[i for i,v in enumerate(wide) if math.isfinite(v)]
result['indexer_scores']={'visible_rows':len(finite),'finite_metrics':compare(array('f',(scalar[i] for i in finite)),array('f',(wide[i] for i in finite))),'cutoff':{str(k):{'scalar':scalar[k],'wide':wide[k]} for k in [3607,1106]}}
for label in ['scalar','wide','corrected','reverse']:
 actual=R/'attention-output'/f'{label}.f32';captured=R/'attention-inputs'/label/'out.f32'
 result['attention'][label]={'capture_exact':actual.read_bytes()==captured.read_bytes(),'sha256':sha(actual),'reference_sha256':sha(captured)}
assert all(v['capture_exact'] for v in result['attention'].values())
result['attention_metrics']=compare(read(R/'attention-output/scalar.f32'),read(R/'attention-output/wide.f32'))
result['topk_by_target']={}
for target in [749,750]:
 result['topk_by_target'][str(target)]={}
 for layer in [20,24]:
  vals={}
  for w in [1,4]:
   d=R/'focused-output'/f'cache-history-w{w}'/f'batch{target//w*w}';ts=list(map(json.loads,(d/'tensors.jsonl').read_text().splitlines()))
   t=next(t for t in ts if t['owner']==f'idx_top_k-{layer}' and t['role']=='output');vals[w]=list(h['read_tensor'](d,t,1,target%w))
  result['topk_by_target'][str(target)][str(layer)]={'ordered_equal':vals[1]==vals[4],'set_equal':set(vals[1])==set(vals[4]),'different_ranks':sum(a!=b for a,b in zip(vals[1],vals[4])),'only_scalar':sorted(set(vals[1])-set(vals[4])),'only_wide':sorted(set(vals[4])-set(vals[1]))}
inputs={}
for name in ['q.f32','weights.f32','mask.f16']:
 a=(R/'indexer-inputs/scalar-w1'/name).read_bytes();b=(R/'indexer-inputs/wide-w1'/name).read_bytes();inputs[name]=a==b
mask=(R/'indexer-inputs/scalar-w1/mask.f16').read_bytes();visible=[i for i,v in enumerate(struct.unpack('<7424e',mask)) if math.isfinite(v)]
keys={w:(R/'indexer-inputs'/f'{k}-w1/k.f16').read_bytes() for w,k in [(1,'scalar'),(4,'wide')]}
inputs['visible_keys_equal']=b''.join(keys[1][i*256:(i+1)*256] for i in visible)==b''.join(keys[4][i*256:(i+1)*256] for i in visible)
inputs['visible_rows']=len(visible);result['indexer_inputs']=inputs
assert inputs['q.f32'] and inputs['mask.f16'] and inputs['visible_keys_equal'] and not inputs['weights.f32']
result['frozen_variants']=result['projection']['variants']+len(result['indexer'])+len(result['attention']);result['all_repeats_stable']=result['projection']['repeat_stable'] and all(v['repeat_bit_exact'] for root in ['indexer-output','attention-output'] for v in map(json.loads,(R/root/'results.jsonl').read_text().splitlines()))
result['scope']='Projection width dependence causally changes top-k at query7367. Repeated-column frozen probes on model CUDA3; no production fix or general numerical compatibility claim.'
(R/'isolated-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
