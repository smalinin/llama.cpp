from pathlib import Path
import importlib.util,json,struct,hashlib
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',R/'analyze-chain.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
manifest={'source':'non-perturbing Stage15 capture after Stage14 callback','datasets':[]}
for target in [0,86]:
 for width in [1,2,4]:
  d=a.D/f'{a.MODE}-w{width}-input{target}'
  ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
  specs={'hidden':('ffn_moe_swiglu_limited-0','output'),'ids':('ffn_moe_topk-0','output'),'weights':('ffn_moe_weights_scaled-0','output'),'captured':('ffn_moe_out-0','output')}
  record={'target':target,'source_width':width,'alignment':json.loads((d/'alignment.json').read_text()),'files':{}}
  for key,(owner,role) in specs.items():
   t=next(t for t in ts if t['owner']==owner and t['role']==role)
   data=a.read_tensor(d,t)
   raw=struct.pack('<'+'i'*len(data),*map(int,data)) if key=='ids' else data.tobytes()
   name=f'input{target}-w{width}-{key}.'+('i32' if key=='ids' else 'f32')
   (R/'inputs'/name).write_bytes(raw)
   record['files'][key]={'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'capture':t}
  manifest['datasets'].append(record)
(R/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
