#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,runpy
R=Path(__file__).resolve().parent;O=R/'attention-inputs';O.mkdir(exist_ok=True)
read_tensor=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['read_tensor'];data={}
for w in [1,4]:
 d=R/'focused-output'/f'cache-history-w{w}'/f'batch{750//w*w}'
 ts={t['role']:t for t in map(json.loads,(d/'tensors.jsonl').read_text().splitlines()) if t['owner']=='FA-24'}
 v={'q.f32':read_tensor(d,ts['src0'],1,750%w).tobytes(),'sinks.f32':read_tensor(d,ts['src4']).tobytes(),'out.f32':read_tensor(d,ts['output'],2,750%w).tobytes(),'op-params.bin':(d/'FA-24-params.bin').read_bytes()}
 for role,name in [('src1','k.f16'),('src2','v.f16')]:
  t=ts[role];raw=(d/t['file']).read_bytes();assert t['ne']==[512,8192,1,1] and t['nb'][0]==2
  v[name]=b''.join(raw[i*t['nb'][1]:i*t['nb'][1]+512*2] for i in range(8192))
 t=ts['src3'];raw=(d/t['file']).read_bytes();v['mask.f16']=raw[(750%w)*t['nb'][1]:(750%w)*t['nb'][1]+8192*2];data[w]=v
summary={}
for label,source,mask_source in [('scalar',1,1),('wide',4,4),('corrected',4,1),('reverse',1,4)]:
 d=O/label;d.mkdir(exist_ok=True);v=dict(data[source]);v['mask.f16']=data[mask_source]['mask.f16'];v['out.f32']=data[mask_source]['out.f32']
 for name,raw in v.items():(d/name).write_bytes(raw)
 (d/'shape.txt').write_text('512 64 8192\n')
 summary[label]={'source_width':source,'mask_source_width':mask_source,'expected_output_width':mask_source,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}}
(R/'attention-input-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
s=(R/'run-indexer.py').read_text().replace("B=R/'indexer-replay'","B=R.parent/'stage18/attention-replay'")
s=s.replace("build=json.loads((R/'indexer-build-manifest.json').read_text());assert build['exit_code']==0 and sha(B.with_suffix('.cpp'))==build['source_sha256']","old=json.loads((R.parent/'stage18/attention-run-manifest.json').read_text());assert sha(B)==old['binary_sha256'] and sha(B.with_suffix('.cpp'))==old['source_sha256']")
s=s.replace("R/'indexer-inputs'","R/'attention-inputs'").replace("R/'indexer-output'","R/'attention-output'").replace("R/'indexer-input-summary.json'","R/'attention-input-summary.json'").replace("R/'indexer.log'","R/'attention.log'").replace("R/'indexer-run-manifest.json'","R/'attention-run-manifest.json'").replace("'indexer exit'","'attention exit'")
(R/'run-attention.py').write_text(s)
