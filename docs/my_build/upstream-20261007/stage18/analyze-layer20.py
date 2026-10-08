from pathlib import Path
from array import array
import json,runpy
R=Path(__file__).resolve().parent
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
s={}
for arch in ['ada','ampere']:
 d=R/f'layer20-{arch}';v={}
 for target in [231,232]:
  scalar=read(d/f't{target}-w1-original.f32');wide=read(d/f't{target}-w3-original.f32')
  crop=read(d/f't{target}-w3-crop.f32');pad=read(d/f't{target}-w1-pad768.f32')
  c={'wide_vs_scalar':compare(scalar,wide),'crop_vs_scalar':compare(scalar,crop),'pad768_vs_wide':compare(wide,pad),'pad768_vs_scalar':compare(scalar,pad)}
  for w in [1,3]:c[f'w{w}_vs_capture']=compare(read(R/'layer20-inputs'/f't{target}-w{w}-original'/'out.f32'),read(d/f't{target}-w{w}-original.f32'))
  assert c['crop_vs_scalar']['bit_identical'] and c['pad768_vs_wide']['bit_identical']
  if arch=='ada':assert all(c[f'w{w}_vs_capture']['bit_identical'] for w in [1,3])
  v[str(target)]=c
 s[arch]=v
(R/'layer20-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
