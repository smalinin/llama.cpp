from pathlib import Path
from array import array
import json,runpy,hashlib
R=Path(__file__).resolve().parent
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
s={'architectures':{}}
for arch in ['ada','ampere']:
 d=R/f'attention-{arch}';v={'native_capture_controls':{},'targets':{}}
 for target in [231,232]:
  base=read(d/f't{target}-w1-original.f32')
  native=read(R/'attention-inputs'/f't{target}-w1-original'/'out.f32')
  v['native_capture_controls'][str(target)]=compare(base,native)
  tv={}
  for w in [1,2,3,4]:
   label=f't{target}-w{w}-original';a=read(d/f'{label}.f32');ref=read(R/'attention-inputs'/label/'out.f32')
   tv[str(w)]={'vs_scalar':compare(base,a),'vs_capture':compare(ref,a)}
   crop=d/f't{target}-w{w}-crop.f32'
   if crop.exists():tv[str(w)]['crop_vs_scalar']=compare(base,read(crop))
  padded=read(d/f't{target}-w1-pad512.f32')
  tv['pad512_vs_scalar']=compare(base,padded)
  wide=3 if target==231 else 2
  tv['pad512_vs_wide']=compare(read(d/f't{target}-w{wide}-original.f32'),padded)
  assert tv['pad512_vs_wide']['bit_identical']
  for w in [2,3,4]:
   if 'crop_vs_scalar' in tv[str(w)]:assert tv[str(w)]['crop_vs_scalar']['bit_identical']
  if arch=='ada':
   assert all(tv[str(w)]['vs_capture']['bit_identical'] for w in [1,2,3,4])
  v['targets'][str(target)]=tv
 s['architectures'][arch]=v
(R/'attention-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
