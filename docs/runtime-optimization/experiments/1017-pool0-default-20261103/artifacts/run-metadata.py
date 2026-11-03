from pathlib import Path
import json,os,subprocess
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(P/'config.json'));args=cfg['command'][1:];base=[];i=0
while i<len(args):
 k=args[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 base.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:base.append(args[i+1]);i+=2
runs=[]
for ctx,flag in [(100352,'0'),(100352,'1'),(176128,'1'),(163840,'1'),(159744,'1')]:
 env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);assert'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in env
 if flag=='0':env['LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL']='0'
 else:assert'LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL'not in env
 env.update(LD_PRELOAD=str(P/'metadata-trace.so'),DSV41_META_TRACE=str(P/f'meta-trace-{ctx}-{flag}.jsonl'));a=base[:];a[a.index('--ctx-size')+1]=str(ctx);cmd=[str(P/'metadata'),str(P/f'metadata-{ctx}-{flag}.json'),'measure',*a]
 with(P/f'metadata-{ctx}-{flag}.log').open('wb')as log:subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.load(open(P/f'metadata-{ctx}-{flag}.json'));assert d['complete']
 runs.append({'context':ctx,'suffix':flag,'pool_env':None,'block_env':None if flag=='1'else'0','command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'fits_512mib':d['fits_512mib']});(P/'metadata-runs.json').write_text(json.dumps({'complete':False,'runs':runs},indent=2)+'\n');print('METADATA',ctx,'default'if flag=='1'else'OFF',[(r['device'],round(r['projected_free']/1048576,2))for r in d['memory']],flush=True)
a,b=[json.load(open(P/f'metadata-100352-{f}.json'))for f in ['0','1']];assert a['routes']==b['routes']
for d in [json.load(open(P/f'metadata-{ctx}-1.json'))for ctx in [176128,163840,159744]]:assert d['routes']==b['routes']
for x,y in zip(a['memory'],b['memory']):
 for k in ['main_model','main_context','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k]
for flag in ['0','1']:
 rows=[json.loads(s)for s in(P/f'meta-trace-100352-{flag}.jsonl').read_text().splitlines()];has_local=any(r['name'].startswith('candidate_local_repeat-')for r in rows);assert has_local==(flag=='1');assert {r['backend']for r in rows if r['name']=='candidate_block_score-20'and r['ne'][1]>6}=={'CUDA3'}
(P/'metadata-runs.json').write_text(json.dumps({'complete':True,'runs':runs,'routes_exact':True,'actual_pool_owner':'CUDA3','source_score_owner':'CUDA2'},indent=2)+'\n');print('METADATA_COMPLETE',flush=True)
