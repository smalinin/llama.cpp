from pathlib import Path
import hashlib,json,math,struct
R=Path(__file__).resolve().parent;D=R/'model-output'
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
O=R/'attention-inputs';O.mkdir(exist_ok=True)
summary={'comparisons':{},'cases':{}}
def column(d,t,axis=None,col=0):
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
def capture(name,target,width,layer):
 start=target//width*width;col=target-start
 d=D/name/f'{MODE}-w{width}-batch{start}'
 ts={t['role']:t for t in map(json.loads,(d/'tensors.jsonl').read_text().splitlines()) if t['owner']==f'FA-{layer}'}
 params=json.loads((d/f'FA-{layer}-params.json').read_text())['op_params_u32']
 data={n:column(d,ts[role],axis,col if axis is not None else 0) for n,role,axis in [('q.f32','src0',1),('k.f16','src1',None),('v.f16','src2',None),('mask.f16','src3',1),('sinks.f32','src4',None),('out.f32','output',2)]}
 mask=list(struct.unpack('<'+'e'*(len(data['mask.f16'])//2),data['mask.f16']))
 visible=[i for i,v in enumerate(mask) if math.isfinite(v)]
 return data,params,visible,mask
def save(label,data,params,metadata):
 d=O/label;d.mkdir(exist_ok=True);nk=len(data['mask.f16'])//2
 for name,raw in data.items():(d/name).write_bytes(raw)
 (d/'shape.txt').write_text(f'512 64 {nk}\n')
 (d/'op-params.bin').write_bytes(struct.pack('<16I',*params))
 summary['cases'][label]={'nk':nk,'n_kv_max':params[4],'sparse_from_predicate':bool(params[4]>0 and nk>=max(4096,2*params[4])),**metadata,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}}
for name,target,width,layer in [('k4096',38,4,20),('ratio1',38,4,20),('ratio2',39,3,2)]:
 a,pa,va,ma=capture(name,target,1,layer);b,pb,vb,mb=capture(name,target,width,layer)
 rows=lambda data,key,indices:b''.join(data[key][i*1024:(i+1)*1024] for i in indices)
 c={'q_bit_exact':a['q.f32']==b['q.f32'],'sinks_bit_exact':a['sinks.f32']==b['sinks.f32'],'params_bit_exact':pa==pb,'visible_indices_equal':va==vb,'visible_mask_equal':[ma[i] for i in va]==[mb[i] for i in vb],'visible_k_bit_exact':rows(a,'k.f16',va)==rows(b,'k.f16',vb),'visible_v_bit_exact':rows(a,'v.f16',va)==rows(b,'v.f16',vb),'native_nk':len(ma),'wide_nk':len(mb)}
 summary['comparisons'][name]=c
 assert all(v for k,v in c.items() if 'exact' in k or 'equal' in k),c
 assert len(mb)==len(ma)+256 and all(v==-math.inf for v in mb[len(ma):])
 meta={'window':name,'input':target,'layer':layer,'capture_output':'post-callback','model_device':0 if layer==2 else 2}
 save(f'{name}-native',a,pa,{**meta,'kind':'native captured shape','expected_capture':True})
 save(f'{name}-wide',b,pb,{**meta,'kind':'wide source shape before diagnostic crop','expected_capture':False})
 crop=dict(b)
 for k in ['k.f16','v.f16']:crop[k]=b[k][:len(ma)*1024]
 crop['mask.f16']=b['mask.f16'][:len(ma)*2]
 save(f'{name}-wide-crop',crop,pb,{**meta,'kind':'wide operands with per-query crop','expected_capture':True})
 pad=dict(a)
 for k in ['k.f16','v.f16']:pad[k]=a[k]+bytes(256*1024)
 pad['mask.f16']=a['mask.f16']+struct.pack('<e',-math.inf)*256
 save(f'{name}-native-pad',pad,pa,{**meta,'kind':'native operands with extra hidden padding','expected_capture':False})
 pd=pa.copy();pd[4]=0
 save(f'{name}-native-dense',a,pd,{**meta,'kind':'native shape with sparse disabled','expected_capture':True})
 save(f'{name}-native-pad-dense',pad,pd,{**meta,'kind':'extra hidden padding with sparse disabled','expected_capture':False})
for name,target,width,layer in [('ratio1',39,4,20),('ratio2',40,3,2)]:
 for w in [1,width]:
  data,params,visible,mask=capture(name,target,w,layer)
  save(f'{name}-first-sparse-w{w}',data,params,{'window':name,'input':target,'layer':layer,'kind':'first native sparse position','model_device':0 if layer==2 else 2,'capture_output':'post-callback','expected_capture':True})
(R/'attention-input-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['comparisons'],indent=2))
