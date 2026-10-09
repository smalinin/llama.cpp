#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,runpy,subprocess
R=Path(__file__).resolve().parent;O=R/'indexer-inputs';O.mkdir(exist_ok=True)
read_tensor=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['read_tensor']
data={}
for w in [1,4]:
 d=R/'focused-output'/f'cache-history-w{w}'/f'batch{750//w*w}';ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
 get=lambda name,role:next(t for t in ts if t['owner']==name and t['role']==role)
 def column(t,axis=None):return read_tensor(d,t,axis,750%w if axis is not None else 0).tobytes()
 k=get('idx_score-24','src1');raw=(d/k['file']).read_bytes();assert k['type']==1 and k['ne']==[128,1,7424,1] and k['nb'][0]==2
 keys=b''.join(raw[i*k['nb'][2]:i*k['nb'][2]+128*2] for i in range(7424))
 mask=get('idx_score-24','src3');raw=(d/mask['file']).read_bytes();maskbytes=raw[(750%w)*mask['nb'][1]:(750%w)*mask['nb'][1]+7424*2]
 data[w]={'q.f32':column(get('idx_score-24','src0'),2),'k.f16':keys,'weights.f32':column(get('idx_score-24','src2'),1),'mask.f16':maskbytes,
 'captured-scores.f32':column(get('idx_score-24','output'),1),'captured-top.i32':(d/get('idx_top_k-24','output')['file']).read_bytes()[(750%w)*2048*4:(750%w+1)*2048*4]}
summary={}
for kind,source,weights in [('scalar',1,1),('wide',4,4),('corrected',4,1),('reverse',1,4)]:
 for width in [1,4]:
  label=f'{kind}-w{width}';dest=O/label;dest.mkdir(exist_ok=True)
  for name,raw in data[source].items():
   if name=='weights.f32':raw=data[weights][name]
   (dest/name).write_bytes(raw*(width if name in ['q.f32','weights.f32','mask.f16'] else 1))
  (dest/'shape.txt').write_text(f'{width} 7424\n')
  summary[label]={'source_width':source,'weight_source_width':weights,'width':width,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}}
(R/'indexer-input-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
S=R.parent/'stage8/candidate-bin';repo=Path('/home/sergei/Github/llama.cpp')
cmd=['g++','-std=c++17','-O2','-I'+str(repo/'ggml/include'),str(R/'indexer-replay.cpp'),'-L'+str(S),'-Wl,-rpath,'+str(S),'-lggml-cuda','-lggml','-lggml-base','-o',str(R/'indexer-replay')]
p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(R/'indexer-build.log').write_text(p.stdout);(R/'indexer-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':p.returncode,'source_sha256':hashlib.sha256((R/'indexer-replay.cpp').read_bytes()).hexdigest()},indent=2)+'\n');print(p.stdout);print('indexer build',p.returncode);raise SystemExit(p.returncode)
