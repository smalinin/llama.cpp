from pathlib import Path
import hashlib,json,math,struct

R=Path(__file__).resolve().parent
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
def save(label,data,params,metadata):
 d=O/label;d.mkdir(exist_ok=True);nk=len(data['mask.f16'])//2
 for name,raw in data.items():(d/name).write_bytes(raw)
 (d/'shape.txt').write_text(f'512 64 {nk}\n')
 (d/'op-params.bin').write_bytes(struct.pack('<16I',*params))
 summary['cases'][label]={'nk':nk,**metadata,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}}
for target,width,layer in [(488,4,0),(489,3,2)]:
 captures={};prefix=f't{target}-fa{layer}'
 for w in [1,width]:
  start=target//w*w;col=target-start
  d=R/'boundary-output/legacy'/f'{MODE}-w{w}-batch{start}'
  ts={t['role']:t for t in map(json.loads,(d/'tensors.jsonl').read_text().splitlines()) if t['owner']==f'FA-{layer}'}
  params=json.loads((d/f'FA-{layer}-params.json').read_text())['op_params_u32']
  data={name:column(d,ts[role],axis,col if axis is not None else 0) for name,role,axis in [('q.f32','src0',1),('k.f16','src1',None),('v.f16','src2',None),('mask.f16','src3',1),('sinks.f32','src4',None),('out.f32','output',2)]}
  mask=list(struct.unpack('<'+'e'*(len(data['mask.f16'])//2),data['mask.f16']));visible=[i for i,v in enumerate(mask) if math.isfinite(v)]
  captures[w]=(data,params,visible,mask)
  save(f'{prefix}-w{w}-original',data,params,{'target':target,'absolute_position':23+target,'width':w,'layer':layer,'kind':'captured-original','visible_rows':visible})
 a,pa,va,ma=captures[1];b,pb,vb,mb=captures[width]
 rows=lambda data,key,indices:b''.join(data[key][i*1024:(i+1)*1024] for i in indices)
 c={'q_bit_exact':a['q.f32']==b['q.f32'],'sinks_bit_exact':a['sinks.f32']==b['sinks.f32'],'params_except_nkvmax_equal':all(x==y for i,(x,y) in enumerate(zip(pa,pb)) if i!=4),'visible_indices_equal':va==vb,'visible_mask_equal':[ma[i] for i in va]==[mb[i] for i in vb],'visible_k_bit_exact':rows(a,'k.f16',va)==rows(b,'k.f16',vb),'visible_v_bit_exact':rows(a,'v.f16',va)==rows(b,'v.f16',vb),'nk1':len(ma),'nkw':len(mb),'native_nkvmax':pa[4],'wide_nkvmax':pb[4]}
 summary['comparisons'][prefix]=c
 assert all(v for k,v in c.items() if 'exact' in k or 'equal' in k)
 assert len(mb)==len(ma)+256 and all(x==-math.inf for x in mb[len(ma):])
 crop=dict(b)
 for key in ['k.f16','v.f16']:crop[key]=b[key][:len(ma)*1024]
 crop['mask.f16']=b['mask.f16'][:len(ma)*2]
 save(f'{prefix}-wide-crop',crop,pb,{'kind':'wide-input-native-padding','expected_output':'native','target':target,'layer':layer})
 padded=dict(a)
 for key in ['k.f16','v.f16']:padded[key]=a[key]+bytes((len(mb)-len(ma))*1024)
 padded['mask.f16']=a['mask.f16']+struct.pack('<e',-math.inf)*(len(mb)-len(ma))
 save(f'{prefix}-native-pad',padded,pa,{'kind':'native-input-extra-masked-padding','expected_output':'wide','target':target,'layer':layer})
 save(f'{prefix}-native-wide-param',a,pb,{'kind':'native-input-wide-nkvmax','expected_output':'native','target':target,'layer':layer})
(R/'attention-input-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['comparisons'],indent=2))
