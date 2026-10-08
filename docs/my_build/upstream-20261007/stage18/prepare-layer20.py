from pathlib import Path
import json,struct,hashlib,math,runpy
R=Path(__file__).resolve().parent
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
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
O=R/'layer20-inputs';O.mkdir(exist_ok=True)
s={'comparisons':{},'cases':{}}
def save(label,data,params,meta):
 d=O/label;d.mkdir(exist_ok=True)
 for name,raw in data.items():(d/name).write_bytes(raw)
 nk=len(data['mask.f16'])//2
 (d/'shape.txt').write_text(f'512 64 {nk}\n');(d/'op-params.bin').write_bytes(struct.pack('<16I',*params))
 s['cases'][label]={'nk':nk,**meta,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}}
for target in [231,232]:
 cap={}
 for w in [1,3]:
  start=target//w*w;col=target-start;d=R/'v2-output/v1-layer20'/f'{MODE}-w{w}-batch{start}'
  ts={t['role']:t for t in map(json.loads,(d/'tensors.jsonl').read_text().splitlines()) if t['owner']=='FA-20'}
  params=json.loads((d/'FA-20-params.json').read_text())['op_params_u32']
  data={name:raw_col(d,ts[role],axis,col if axis is not None else 0) for name,role,axis in [('q.f32','src0',1),('k.f16','src1',None),('v.f16','src2',None),('mask.f16','src3',1),('sinks.f32','src4',None),('out.f32','output',2)]}
  # The v1 callback already removes the raw gap before this captured output.
  original_sha={k:hashlib.sha256(v).hexdigest() for k,v in data.items()}
  if w==3:
   mask=struct.unpack('<'+'e'*(len(data['mask.f16'])//2),data['mask.f16'])
   assert all(not math.isfinite(x) for x in mask[256:512])
   for k in ['k.f16','v.f16']:data[k]=data[k][:256*1024]+data[k][512*1024:]
   data['mask.f16']=data['mask.f16'][:256*2]+data['mask.f16'][512*2:]
  mask=struct.unpack('<'+'e'*(len(data['mask.f16'])//2),data['mask.f16']);visible=[i for i,v in enumerate(mask) if math.isfinite(v)]
  cap[w]=(data,params,visible,mask)
  save(f't{target}-w{w}-original',data,params,{'kind':'native' if w==1 else 'v1-raw-crop','original_capture_column_sha256':original_sha,'column':col,'target':target})
 a,pa,va,ma=cap[1];b,pb,vb,mb=cap[3]
 rows=lambda data,key,ix:b''.join(data[key][i*1024:(i+1)*1024] for i in ix)
 c={'q_bit_exact':a['q.f32']==b['q.f32'],'sinks_bit_exact':a['sinks.f32']==b['sinks.f32'],'params_bit_exact':pa==pb,'scale_bias_cap_precision_bit_exact':pa[:4]==pb[:4],'native_n_kv_max':pa[4],'wide_n_kv_max':pb[4],'visible_indices_equal':va==vb,'visible_mask_values_equal':[ma[i] for i in va]==[mb[i] for i in vb],'visible_k_bit_exact':rows(a,'k.f16',va)==rows(b,'k.f16',vb),'visible_v_bit_exact':rows(a,'v.f16',va)==rows(b,'v.f16',vb),'nk1':len(ma),'nkw3_after_raw_crop':len(mb)}
 s['comparisons'][str(target)]=c
 assert all(v for k,v in c.items() if ('exact' in k or 'equal' in k) and k!='params_bit_exact'),c
 assert len(ma)==512 and len(mb)==768
 assert all(not math.isfinite(v) for v in mb[512:])
 crop=dict(b)
 for k in ['k.f16','v.f16']:crop[k]=b[k][:512*1024]
 crop['mask.f16']=b['mask.f16'][:512*2]
 save(f't{target}-w3-crop',crop,pb,{'kind':'restore-native-compressed-padding','target':target})
 pad=dict(a)
 for k in ['k.f16','v.f16']:pad[k]=a[k]+bytes(256*1024)
 pad['mask.f16']=a['mask.f16']+struct.pack('<e',-math.inf)*256
 save(f't{target}-w1-pad768',pad,pa,{'kind':'scalar-extra-masked-compressed-padding','target':target})
(R/'layer20-input-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s['comparisons'],indent=2))
