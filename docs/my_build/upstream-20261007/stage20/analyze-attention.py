from pathlib import Path
from array import array
import json,runpy
R=Path(__file__).resolve().parent
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
meta=json.loads((R/'attention-input-summary.json').read_text())
summary={'architectures':{}}
for arch,index in [('ada0',0),('ada2',2),('ampere',5)]:
 D=R/f'attention-{arch}';s={'windows':{},'model_capture_controls':{}}
 for name in ['k4096','ratio1','ratio2']:
  native=read(D/f'{name}-native.f32');wide=read(D/f'{name}-wide.f32')
  c={'uncropped_wide_vs_native':compare(native,wide),'crop_vs_native':compare(native,read(D/f'{name}-wide-crop.f32')),'pad_vs_wide':compare(wide,read(D/f'{name}-native-pad.f32')),'native_dense_vs_native':compare(native,read(D/f'{name}-native-dense.f32')),'pad_sparse_vs_pad_dense':compare(read(D/f'{name}-native-pad.f32'),read(D/f'{name}-native-pad-dense.f32'))}
  assert all(c[k]['bit_identical'] for k in ['crop_vs_native','pad_vs_wide','native_dense_vs_native'])
  s['windows'][name]=c
 for label,item in meta['cases'].items():
  if item['expected_capture'] and item['model_device']==index:
   c=compare(read(D/f'{label}.f32'),read(R/'attention-inputs'/label/'out.f32'))
   assert c['bit_identical'],(arch,label,c)
   s['model_capture_controls'][label]=c
 summary['architectures'][arch]=s
(R/'attention-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
