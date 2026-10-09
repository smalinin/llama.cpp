#!/usr/bin/env python3
from pathlib import Path
from array import array
import ctypes,hashlib,json,math,runpy
R=Path(__file__).resolve().parent
D=R/'focused-output'
h=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))
read_tensor,metric=h['read_tensor'],h['metric']
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def lines(path):return [json.loads(l) for l in path.read_text().splitlines()]
def compare(a,b):
 assert len(a)==len(b)
 finite_a=array('f');finite_b=array('f');nonfinite=[]
 for i,(u,v) in enumerate(zip(a,b)):
  if math.isfinite(u) and math.isfinite(v):finite_a.append(u);finite_b.append(v)
  elif u!=v:nonfinite.append(i)
 out=(ctypes.c_double*5)()
 if finite_a:metric(finite_a.buffer_info()[0],finite_b.buffer_info()[0],len(finite_a),0,out)
 assert out[4]==0
 return {'bit_identical':a.tobytes()==b.tobytes(),'max_abs_finite':out[2],'rms_finite':math.sqrt(out[0]/len(finite_a)) if finite_a else None,'nonfinite_mismatch_count':len(nonfinite),'nonfinite_mismatch_first':nonfinite[:10]}
result={'controls':{},'fresh_capture_controls':{},'targets':{}}
for w in [1,4]:
 actual=sha(D/f'cache-history-w{w}/logits.f32');reference=sha(R.parent/f'stage22/model-output/cache-history-w{w}/logits.f32')
 result['controls'][str(w)]={'sha256':actual,'reference_sha256':reference,'bit_identical':actual==reference}
 actual=sha(D/f'fresh-control-w{w}/logits.f32');reference=sha(R/f'model-output/fresh-history-w{w}/logits.f32')
 result['fresh_capture_controls'][str(w)]={'sha256':actual,'reference_sha256':reference,'bit_identical':actual==reference}
assert all(v['bit_identical'] for family in [result['controls'],result['fresh_capture_controls']] for v in family.values()),'capture changed logits'
for target in [749,750]:
 dirs={w:D/f'cache-history-w{w}'/f'batch{target//w*w}' for w in [1,4]}
 tensors={w:lines(d/'tensors.jsonl') for w,d in dirs.items()}
 chain={w:[t for t in ts if t['role']=='output'] for w,ts in tensors.items()}
 assert len(chain[1])==len(chain[4]);records=[];occ={}
 for a,b in zip(chain[1],chain[4]):
  assert a['owner']==b['owner'];key=a['owner'];occurrence=occ.get(key,0);occ[key]=occurrence+1
  axes=[j for j,(u,v) in enumerate(zip(a['ne'],b['ne'])) if u!=v];assert len(axes)<=1,(a,b)
  axis=axes[0] if axes else None
  aa=read_tensor(dirs[1],a,axis,0);bb=read_tensor(dirs[4],b,axis,target%4 if axis is not None else 0)
  v=compare(aa,bb);v.update(owner=key,occurrence=occurrence,op=a['op'],ne1=a['ne'],ne4=b['ne'],axis=axis,elements=len(aa),file1=a['file'],file4=b['file']);records.append(v)
 fa={w:{t['role']:t for t in ts if t['owner']=='FA-24'} for w,ts in tensors.items()}
 masks={w:read_tensor(dirs[w],fa[w]['src3'],1,target%w) for w in [1,4]}
 visible={w:[i for i,v in enumerate(m) if math.isfinite(v)] for w,m in masks.items()}
 common=sorted(set(visible[1])&set(visible[4]));only1=sorted(set(visible[1])-set(visible[4]));only4=sorted(set(visible[4])-set(visible[1]))
 def kv_rows(w,role):
  t=fa[w][role];raw=(dirs[w]/t['file']).read_bytes();assert t['nb'][0]==2 and t['ne'][2:]==[1,1]
  return b''.join(raw[i*t['nb'][1]:i*t['nb'][1]+t['ne'][0]*2] for i in common)
 checks={'q_bit_exact':read_tensor(dirs[1],fa[1]['src0'],1,0).tobytes()==read_tensor(dirs[4],fa[4]['src0'],1,target%4).tobytes(),
 'sinks_bit_exact':read_tensor(dirs[1],fa[1]['src4']).tobytes()==read_tensor(dirs[4],fa[4]['src4']).tobytes(),
 'op_params_bit_exact':(dirs[1]/'FA-24-params.bin').read_bytes()==(dirs[4]/'FA-24-params.bin').read_bytes(),
 'mask_bit_exact':masks[1].tobytes()==masks[4].tobytes(),'visible_count':{w:len(v) for w,v in visible.items()},'only_scalar_visible':only1,'only_wide_visible':only4,
 'common_visible_k_bit_exact':kv_rows(1,'src1')==kv_rows(4,'src1'),'common_visible_v_bit_exact':kv_rows(1,'src2')==kv_rows(4,'src2')}
 result['targets'][str(target)]={'chain':records,'first_differences':[r for r in records if not r['bit_identical']][:20],'attention_inputs':checks}
(R/'focused-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in result.items() if k!='targets'},'targets':{k:{n:v for n,v in x.items() if n!='chain'} for k,x in result['targets'].items()}},indent=2))
