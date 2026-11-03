from pathlib import Path
import os,json,subprocess
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(P/'config.json'));args=cfg['command'][1:];trim=[];i=0
while i<len(args):
 k=args[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 trim.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:trim.append(args[i+1]);i+=2
for ctx,flag in [(100352,'0'),(100352,'1'),(176128,'0'),(176128,'1')]:
 e={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','DSV','CUDA_'))and k not in['LD_PRELOAD','LD_LIBRARY_PATH']};e.update(cfg['environment']);e.update(LLAMA_DSV41_CANDIDATE_POOL_LOCAL=flag,LD_PRELOAD=str(P/'metadata-trace.so'),DSV41_META_TRACE=str(P/f'meta-trace-{ctx}-{flag}.jsonl'));a=trim[:];a[a.index('--ctx-size')+1]=str(ctx)
 with(P/f'metadata-{ctx}-{flag}.log').open('wb')as log:subprocess.run(['/tmp/dsv41-hc-on-long-indexer-20261103/metadata',str(P/f'metadata-{ctx}-{flag}.json'),'measure',*a],cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,check=True)
 x=json.load(open(P/f'metadata-{ctx}-{flag}.json'));assert x['complete']and x['fits_512mib'];print('METADATA',ctx,flag,[(q['device'],round(q['projected_free']/1048576,2))for q in x['memory']],flush=True)
