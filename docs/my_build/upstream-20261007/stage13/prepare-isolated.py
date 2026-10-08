#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',ROOT/'analyze-chain.py');chain=importlib.util.module_from_spec(spec);spec.loader.exec_module(chain)
spec=importlib.util.spec_from_file_location('inspection',ROOT.parent/'stage7/inspect-models.py');inspect=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspect)
model=Path(json.loads((ROOT.parent/'stage7/model-inspection.json').read_text())['models']['model']['path'])
OUT=ROOT/'inputs';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index={}
for path in sorted(model.parent.glob('*.gguf')):
 with path.open('rb') as f:
  magic,version,count,keys=struct.unpack('<4sIQQ',f.read(24));assert magic==b'GGUF' and version==3
  metadata={}
  for _ in range(keys):
   key=inspect.string(f);metadata[key]=inspect.value(f,inspect.u32(f))
  tensors=[]
  for _ in range(count):
   name=inspect.string(f);dims=[inspect.u64(f) for _ in range(inspect.u32(f))];kind,offset=inspect.u32(f),inspect.u64(f)
   tensors.append((name,dims,kind,offset))
  alignment=metadata.get('general.alignment',32);start=(f.tell()+alignment-1)//alignment*alignment
  for name,dims,kind,offset in tensors:
   if name not in ['blk.0.ffn_gate_inp.weight','blk.0.hc_attn_fn.weight']:continue
   assert kind in [0,30]
   size=math.prod(dims)*(4 if kind==0 else 2);f.seek(start+offset);raw=f.read(size);assert len(raw)==size
   key='router0' if 'ffn_gate' in name else 'hc0'
   p=OUT/key;p.mkdir(exist_ok=True);(p/'weight.bin').write_bytes(raw)
   index[key]={'source':str(path),'tensor':name,'type':kind,'shape':dims,'offset':start+offset,'bytes':size,'sha256':sha(p/'weight.bin'),'model_size':path.stat().st_size,'model_mtime_ns':path.stat().st_mtime_ns}
assert set(index)=={'router0','hc0'}
mode='decode-scalar-fa-upgate-boundaries'
for width in [1,2,4]:
 d=ROOT/'chain-capture-output'/f'{mode}-w{width}';ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
 get=lambda owner,role:next(t for t in ts if t['owner']==owner and t['role']==role)
 for key in ['hc0','router0']:
  inp=get('hc_mixes-0','src1') if key=='hc0' else get('ffn_norm-0','output')
  out=get('hc_mixes-0','output') if key=='hc0' else get('ffn_moe_logits-0','output')
  p=OUT/key
  (p/f'input-w{width}.f32').write_bytes(chain.read_tensor(d,inp).tobytes())
  (p/f'captured-w{width}.f32').write_bytes(chain.read_tensor(d,out).tobytes())
  if key=='hc0':assert (d/get('hc_mixes-0','src0')['file']).read_bytes()==(p/'weight.bin').read_bytes()
 for layer in [0,2]:
  key=f'post{layer}';p=OUT/key;p.mkdir(exist_ok=True)
  for role in ['src0','src1','src2','src3','output']:
   t=get(f'hc_attn_post-{layer}',role)
   (p/f'w{width}-{role}.f32').write_bytes(chain.read_tensor(d,t).tobytes())
  meta=index.setdefault(key,{'source':'chain-capture first decode HC attention post operands','layer':layer,'shapes_by_width':{}})
  meta['shapes_by_width'][str(width)]={'src0':[5120,width],'src1':[5120,4,width],'src2':[4,width],'src3':[4,4,width],'output':[5120,4,width]}
for key,meta in index.items():
 meta['files']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((OUT/key).glob('*')) if p.name not in ['manifest.json','config.txt']}
 (OUT/key/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
(OUT/'manifest.json').write_text(json.dumps(index,indent=2)+'\n')
for key in ['hc0','router0']:
 meta=index[key];(OUT/key/'config.txt').write_text(f"{meta['type']} {meta['shape'][0]} {meta['shape'][1]}\n")
print(json.dumps({k:{'weight_type':v.get('type'),'weight_shape':v.get('shape'),'files':len(v['files'])} for k,v in index.items()}))
