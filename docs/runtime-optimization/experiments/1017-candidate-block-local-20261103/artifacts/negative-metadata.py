from pathlib import Path
import os,json,subprocess
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(P/'config.json'));args=cfg['command'][1:];base=[];i=0
while i<len(args):
 k=args[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 base.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:base.append(args[i+1]);i+=2
base[base.index('--ctx-size')+1]='32768'
j={'complete':False,'scope':'Production graph fallback in fresh no_alloc processes; no real-model quality/speed claim for unsupported modes.','cases':[]}
for label,key,value in [('parallel2','--parallel','2'),('topk1024','--override-kv','deepseek41.attention.indexer.top_k=int:1024')]:
 arms=[]
 for flag in ['0','1']:
  a=base[:];a[a.index(key)+1]=value;trace=P/f'negative-{label}-{flag}.jsonl';env={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','DSV','CUDA_'))and k not in['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(cfg['environment']);env.update(LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=flag,LD_PRELOAD=str(P/'metadata-trace.so'),DSV41_META_TRACE=str(trace))
  out=P/f'negative-{label}-{flag}.json'
  with(P/f'negative-{label}-{flag}.log').open('wb')as log:subprocess.run([str(P/'metadata'),str(out),'measure',*a],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  d=json.load(open(out));assert d['complete'];rows=[json.loads(l)for l in trace.read_text().splitlines()];assert not any('candidate_local_'in r['name']for r in rows)
  arms.append((d,{(r['name'],r['op'],r['backend'],tuple(r['ne']))for r in rows}))
 assert arms[0][0]['routes']==arms[1][0]['routes']and arms[0][1]==arms[1][1]
 for x,y in zip(arms[0][0]['memory'],arms[1][0]['memory']):
  for k in ['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k],(label,k)
 j['cases'].append({'label':label,'fallback_graph_and_budgets_exact':True})
pool=[]
for flag in ['0','1']:
 a=base[:];a[a.index('--ctx-size')+1]='32768';env={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','DSV','CUDA_'))and k not in['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(cfg['environment']);trace=P/f'pool0-block{flag}.jsonl';env.update(LLAMA_DSV41_CANDIDATE_POOL_LOCAL='0',LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=flag,LD_PRELOAD=str(P/'metadata-trace.so'),DSV41_META_TRACE=str(trace));out=P/f'pool0-block{flag}.json'
 with(P/f'pool0-block{flag}.log').open('wb')as log:subprocess.run([str(P/'metadata'),str(out),'measure',*a],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.load(open(out));assert d['complete'];rows=[json.loads(l)for l in trace.read_text().splitlines()];local=sorted({(r['name'],r['backend'])for r in rows if 'candidate_local_repeat' in r['name']});assert bool(local)==(flag=='1');assert {r['backend']for r in rows if r['name']=='candidate_block_score-20'and r['ne'][1]>6}=={'CUDA3'}
 pool.append({'flag':flag,'routes':d['routes'],'local':local})
assert pool[0]['routes']==pool[1]['routes'];j['pool0_metadata_control']=[{'flag':x['flag'],'local_repeats':x['local']}for x in pool]
j['complete']=True;(P/'negative-results.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
