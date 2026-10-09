#!/usr/bin/env python3
from pathlib import Path
from array import array
import hashlib,importlib.util,json,math,runpy,struct
R=Path(__file__).resolve().parent;OUT=R/'projection-inputs';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('inspection',R.parent/'stage7/inspect-models.py');inspection=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspection)
read_tensor=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['read_tensor']
model=Path(json.loads((R.parent/'stage7/model-inspection.json').read_text())['models']['model']['path'])
found=[]
for path in sorted(model.parent.glob('*.gguf')):
 with path.open('rb') as f:
  magic,version,count,keys=struct.unpack('<4sIQQ',f.read(24));assert magic==b'GGUF' and version==3
  metadata={}
  for _ in range(keys):
   key=inspection.string(f);metadata[key]=inspection.value(f,inspection.u32(f))
  tensors=[]
  for _ in range(count):
   name=inspection.string(f);dims=[inspection.u64(f) for _ in range(inspection.u32(f))];kind,offset=inspection.u32(f),inspection.u64(f);tensors.append((name,dims,kind,offset))
  alignment=metadata.get('general.alignment',32);start=(f.tell()+alignment-1)//alignment*alignment
  for name,dims,kind,offset in tensors:
   if name!='blk.24.indexer.proj.weight':continue
   assert dims==[5120,32] and kind in [0,30]
   size=math.prod(dims)*(4 if kind==0 else 2);f.seek(start+offset);raw=f.read(size);assert len(raw)==size
   (OUT/'weight.bin').write_bytes(raw)
   found.append({'source':str(path),'tensor':name,'shape':dims,'type':kind,'offset':start+offset,'bytes':size,'sha256':hashlib.sha256(raw).hexdigest(),'model_size':path.stat().st_size,'model_mtime_ns':path.stat().st_mtime_ns})
assert len(found)==1
meta=found[0];(OUT/'config.txt').write_text(f"{meta['type']} 5120 32\n")
for w in [1,4]:
 d=R/'focused-output'/f'cache-history-w{w}'/f'batch{750//w*w}'
 ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
 get=lambda n:next(t for t in ts if t['owner']==n and t['role']=='output')
 inputs=read_tensor(d,get('attn_norm-24'));outputs=read_tensor(d,get('idx_weights-24'))
 (OUT/f'input-w{w}.f32').write_bytes(inputs.tobytes());(OUT/f'captured-w{w}.f32').write_bytes(outputs.tobytes())
 if w==1:
  (OUT/'input-w2.f32').write_bytes(inputs.tobytes()*2)
  (OUT/'captured-w2.f32').write_bytes(outputs.tobytes()*2)
scalar=array('f');scalar.frombytes((OUT/'input-w1.f32').read_bytes());wide=array('f');wide.frombytes((OUT/'input-w4.f32').read_bytes())
assert scalar.tobytes()==wide[2*5120:3*5120].tobytes()
meta.update(target_input_index=750,query_position=7367,wide_column=2,score_scale=1/64,source2='scalar input repeated twice',input_columns_exact=True)
meta['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='files'},indent=2))
