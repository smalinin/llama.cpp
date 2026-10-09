from pathlib import Path
from array import array
import hashlib,json,re,math
R=Path(__file__).resolve().parent;root=R/'server-runs';ref=root/'trace-native/glm5next-spec-0'
def compare(d,baseline):
 out=[]
 for f in sorted(d.glob('*-response.json')):
  if not (baseline/f.name).exists():continue
  a=json.loads(f.read_text());b=json.loads((baseline/f.name).read_text());out.append({'file':f.name,'tokens_equal':a['tokens']==b['tokens'],'text_equal':a['content']==b['content'],'probabilities_equal':a.get('completion_probabilities')==b.get('completion_probabilities')})
 return out
controls={label:compare(root/label/'glm5next-spec-0',ref) for label in ['roundtrip-native','fa-native','capture-native']}
controls['stage29_native_vs_trace']=compare(ref,R.parent/'stage29/server-runs/native-ram-off/glm5next-spec-0')
lines=[x for x in (root/'roundtrip-native/glm5next-spec-0/server.log').read_text().splitlines() if 'STAGE30_ROUNDTRIP' in x]
layouts=[x for x in (ref/'server.log').read_text().splitlines() if 'STAGE30_LAYOUT' in x]
obj={'controls':controls,'roundtrip_count':len(lines),'all_roundtrips_byte_exact':all(x.endswith('equal=1') for x in lines),'roundtrip_log_lines':lines,'layout_checks':len(layouts),'layout_mismatches':sum('mismatch=0 ' not in x for x in layouts),'broad_capture_accepted_as_baseline':False,'narrow_fa_capture_accepted_as_baseline':len(controls['fa-native'])==12 and all(x['tokens_equal'] and x['probabilities_equal'] for x in controls['fa-native'])}
fa=[]
first=array('f',(R/'fa-native-tensors/0-fa.bin').read_bytes())
for i in [1,2,3]:
 cur=array('f',(R/f'fa-native-tensors/{i}-fa.bin').read_bytes());delta=[float(x)-y for x,y in zip(first,cur)];fa.append({'capture':i,'byte_exact':first.tobytes()==cur.tobytes(),'max_abs':max(map(abs,delta)),'rms':math.sqrt(sum(x*x for x in delta)/len(delta))})
obj['first_fa_output_differences']=fa
cpu=[]
for name in ['cpu/before-native','cpu/occupied-before','cpu/state-before','gpu/tiny-before']:
 d=R/name
 for seq in [2,1,0]:
  a=array('f',(d/f'phase0-seq{seq}.f32').read_bytes())
  for phase in [1,2,3]:
   b=array('f',(d/f'phase{phase}-seq{seq}.f32').read_bytes());delta=[float(x)-y for x,y in zip(a,b)];cpu.append({'case':name,'seq':seq,'phase':phase,'byte_exact':a.tobytes()==b.tobytes(),'max_abs':max(map(abs,delta)),'rms':math.sqrt(sum(x*x for x in delta)/len(delta)),'changed_argmax_rows':sum(max(range(128),key=lambda j:a[i+j])!=max(range(128),key=lambda j:b[i+j]) for i in range(0,len(a),128))})
obj['tiny_fixture_comparisons']=cpu
(R/'diagnostic-controls.json').write_text(json.dumps(obj,indent=2)+'\n')
assert obj['all_roundtrips_byte_exact'] and obj['layout_mismatches']==0 and obj['narrow_fa_capture_accepted_as_baseline']
assert all(x['tokens_equal'] and x['probabilities_equal'] for x in controls['stage29_native_vs_trace'])
print('diagnostic controls verified')
