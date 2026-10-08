from pathlib import Path
import json,struct,runpy,hashlib,math
R=Path(__file__).resolve().parent
H=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'));read=H['read_tensor'];compare=H['compare']
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
O=R/'attention-inputs';O.mkdir(exist_ok=True)
summary={'comparisons':{},'cases':{}}
def raw_col(d,t,axis=None,col=0):
 raw=(d/t['file']).read_bytes();ne=t['ne'].copy();off=0
 if axis is not None:ne[axis]=1;off=col*t['nb'][axis]
 size={0:4,1:2}[t['type']];out=bytearray()
 for i3 in range(ne[3]):
  for i2 in range(ne[2]):
   for i1 in range(ne[1]):
    start=off+i1*t['nb'][1]+i2*t['nb'][2]+i3*t['nb'][3]
    assert t['nb'][0]==size
    out.extend(raw[start:start+ne[0]*size])
 return bytes(out)
def save(label,data,params,metadata):
 d=O/label;d.mkdir(exist_ok=True);nk=len(data['mask.f16'])//2
 for name,raw in data.items():(d/name).write_bytes(raw)
 (d/'shape.txt').write_text(f'512 64 {nk}\n')
 (d/'op-params.bin').write_bytes(struct.pack('<16I',*params))
 summary['cases'][label]={'nk':nk,**metadata,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}}
for target in [231,232]:
 captures={}
 for w in [1,2,3,4]:
  start=target//w*w;col=target-start;d=R/'capture-output'/f'{MODE}-w{w}-batch{start}'
  ts={t['role']:t for t in map(json.loads,(d/'tensors.jsonl').read_text().splitlines()) if t['owner']=='FA-0'}
  params=json.loads((d/'FA-0-params.json').read_text())['op_params_u32']
  data={name:raw_col(d,ts[role],axis,col if axis is not None else 0) for name,role,axis in [('q.f32','src0',1),('k.f16','src1',None),('v.f16','src2',None),('mask.f16','src3',1),('sinks.f32','src4',None),('out.f32','output',2)]}
  mask=list(struct.unpack('<'+'e'*(len(data['mask.f16'])//2),data['mask.f16']));visible=[i for i,v in enumerate(mask) if math.isfinite(v)]
  captures[w]=(data,params,visible,mask)
  label=f't{target}-w{w}-original';save(label,data,params,{'target':target,'width':w,'column':col,'batch_start':start,'kind':'captured-original','visible_rows':visible})
 a,pa,va,ma=captures[1]
 summary['comparisons'][str(target)]={}
 for w in [2,3,4]:
  b,pb,vb,mb=captures[w];assert pa==pb
  rows=lambda data,key,indices:b''.join(data[key][i*1024:(i+1)*1024] for i in indices)
  c={'q_bit_exact':a['q.f32']==b['q.f32'],'sinks_bit_exact':a['sinks.f32']==b['sinks.f32'],'params_bit_exact':pa==pb,'visible_indices_equal':va==vb,'visible_mask_values_equal':[ma[i] for i in va]==[mb[i] for i in vb], 'visible_k_bit_exact':rows(a,'k.f16',va)==rows(b,'k.f16',vb),'visible_v_bit_exact':rows(a,'v.f16',va)==rows(b,'v.f16',vb),'nk1':len(ma),'nkw':len(mb)}
  summary['comparisons'][str(target)][str(w)]=c
  assert all(v for k,v in c.items() if 'exact' in k or 'equal' in k)
  if len(mb)>len(ma):
   assert all(not math.isfinite(x) for x in mb[len(ma):])
   crop=dict(b)
   for key in ['k.f16','v.f16']:crop[key]=b[key][:len(ma)*1024]
   crop['mask.f16']=b['mask.f16'][:len(ma)*2]
   save(f't{target}-w{w}-crop',crop,pb,{'kind':'wide-input-native-padding','target':target,'width':w,'expected_output':'scalar'})
 padded=dict(a)
 for key in ['k.f16','v.f16']:padded[key]=a[key]+bytes((512-len(ma))*1024)
 padded['mask.f16']=a['mask.f16']+struct.pack('<e',-math.inf)*(512-len(ma))
 save(f't{target}-w1-pad512',padded,pa,{'kind':'scalar-input-extra-masked-padding','target':target,'width':1,'expected_output':'wide'})
(R/'attention-input-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['comparisons'],indent=2))
