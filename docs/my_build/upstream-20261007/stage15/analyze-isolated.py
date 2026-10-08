from array import array
import ctypes,importlib.util,json
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',R/'analyze-chain.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
lib=ctypes.CDLL(str(R/'cpu-reduction.so'));reduce=lib.stage15_reduce
reduce.argtypes=[ctypes.c_void_p]*4+[ctypes.c_size_t,ctypes.c_int]
def load(p):
 v=array('f');v.frombytes(p.read_bytes());return v
def model(d,w):
 x=array('f',[0])*5120;y=array('f',[0])*5120
 reduce(d.buffer_info()[0],w.buffer_info()[0],x.buffer_info()[0],y.buffer_info()[0],5120,6)
 return x,y
summary={'native_cases':44,'component_cases':88,'capture_controls':[],'fixed_operands':{},'decomposition':[],'component_capture_comparisons_performed':False}
for arch in ['ada','ampere']:
 rs=[json.loads(l) for l in (R/f'down0-{arch}/results.jsonl').read_text().splitlines()]
 assert len(rs)==22
 for x in rs:
  assert x['repeat_stable'] and x['column_max']==0
  if x['actual'] or (x['source']==1 and x['width']==1):
   assert x['max_vs_capture']==0
   summary['capture_controls'].append({'architecture':arch,'label':x['label'],'max_abs':0})
 summary['fixed_operands'][arch]={str(t):max(x['max_vs_scalar'] for x in rs if x['target']==t) for t in [0,86]}
 for target in [0,86]:
  weights=load(R/f'inputs/input{target}-w1-weights.f32')
  ds={w:load(R/f'down-{arch}/input{target}-source1-w{w}-repeat.f32')[:5120*6] for w in [1,2,4]}
  raw={w:load(R/f'down0-{arch}/input{target}-source1-w{w}-repeat.f32')[:5120] for w in [1,2,4]}
  round1,fma1=model(ds[1],weights)
  round2,fma2=model(ds[2],weights)
  round4,fma4=model(ds[4],weights)
  r={'architecture':arch,'target':target,
     'scalar_rounded_cpu_vs_native':a.compare(raw[1],round1),
     'wide2_fma_cpu_vs_native':a.compare(raw[2],fma2),
     'wide4_fma_cpu_vs_native':a.compare(raw[4],fma4),
     'down_width2_vs1':a.compare(ds[1],ds[2]),
     'down_width4_vs2':a.compare(ds[2],ds[4]),
     'down_effect_with_rounded_sum':a.compare(round1,round2),
     'fma_effect_on_wide_down':a.compare(round2,fma2),
     'combined_native_difference':a.compare(raw[1],raw[2])}
  assert all(r[k]['bit_identical'] for k in ['scalar_rounded_cpu_vs_native','wide2_fma_cpu_vs_native','wide4_fma_cpu_vs_native','down_width4_vs2'])
  summary['decomposition'].append(r)
assert len(summary['capture_controls'])==12
(R/'isolated-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
