from pathlib import Path
from array import array
import json,runpy
R=Path(__file__).resolve().parent
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
s={'architectures':{}}
for arch in ['ada','ampere']:
 d=R/f'attention-{arch}';v={}
 for target,width,layer in [(488,4,0),(489,3,2)]:
  prefix=f't{target}-fa{layer}'
  native=read(d/f'{prefix}-w1-original.f32');wide=read(d/f'{prefix}-w{width}-original.f32')
  tv={'wide_vs_native':compare(native,wide),'native_capture':compare(native,read(R/'attention-inputs'/f'{prefix}-w1-original/out.f32')),'wide_capture':compare(wide,read(R/'attention-inputs'/f'{prefix}-w{width}-original/out.f32')),
      'crop_vs_native':compare(native,read(d/f'{prefix}-wide-crop.f32')),'pad_vs_wide':compare(wide,read(d/f'{prefix}-native-pad.f32')),'wide_param_vs_native':compare(native,read(d/f'{prefix}-native-wide-param.f32'))}
  assert all(tv[k]['bit_identical'] for k in ['crop_vs_native','pad_vs_wide','wide_param_vs_native'])
  if arch=='ada':assert tv['native_capture']['bit_identical'] and tv['wide_capture']['bit_identical']
  v[prefix]=tv
 s['architectures'][arch]=v
(R/'attention-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
