from pathlib import Path
import json,os,subprocess,hashlib,difflib
P=Path(__file__).parent;O=Path('/tmp/dsv41-candidate-block-local-20261103');R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(O/'config.json'));args=cfg['command'][1:];base=[];i=0
while i<len(args):
 k=args[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 base.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:base.append(args[i+1]);i+=2
j={'complete':False,'scope':'Default promotion only: fresh production no_alloc unset vs explicit1 graph/budget/routes equality, explicit0 fallback. Pool0 and pool1 checked; prior explicit1 native gates are historical and not rerun. Internal headers/helper ABI unchanged.','cases':[],'libraries':{n:hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()for n in cfg['source_library_hashes']},'helper_hashes':{n:hashlib.sha256((O/n).read_bytes()).hexdigest()for n in ['metadata','metadata-trace.so']}}
models={}
for ctx,pool,flags in [(100352,'0',['default','1','0']),(100352,'1',['default','1']),(176128,'0',['default','1'])]:
 results={}
 for flag in flags:
  tag=f'{ctx}-pool{pool}-{flag}';env={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.pop('LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL',None);env.update(LLAMA_DSV41_CANDIDATE_POOL_LOCAL=pool,LD_PRELOAD=str(O/'metadata-trace.so'),DSV41_META_TRACE=str(P/(tag+'.jsonl')))
  if flag!='default':env['LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL']=flag
  a=base[:];a[a.index('--ctx-size')+1]=str(ctx);cmd=[str(O/'metadata'),str(P/(tag+'.json')),'measure',*a]
  with(P/(tag+'.log')).open('wb')as log:subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  m=json.load(open(P/(tag+'.json')));assert m['complete'];rows=[json.loads(s)for s in(P/(tag+'.jsonl')).read_text().splitlines()];placement={(r['name'],r['op'],r['backend'],tuple(r['ne']))for r in rows};local={r for r in placement if r[0].startswith('candidate_local_repeat-')};assert bool(local)==(flag!='0')
  if flag!='0':assert {x[2]for x in local}=={'CUDA3','CUDA4'}and not any(x[0]=='candidate_mask-20'and x[3][1]>6 for x in placement)
  text=(P/(tag+'.log')).read_text(errors='replace');assert 'does not match expectation'not in text
  j['cases'].append({'context':ctx,'pool':pool,'flag':flag,'command':cmd,'environment_delta':{'LLAMA_DSV41_CANDIDATE_POOL_LOCAL':pool,'LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL':None if flag=='default'else flag},'local_repeats':len(local),'fits_512mib':m['fits_512mib']});results[flag]=(m,placement)
  (P/'results.json').write_text(json.dumps(j,indent=2)+'\n');print('PASS',tag,'local',len(local),flush=True)
 a,b=results['default'],results['1'];assert a[1]==b[1]and a[0]['routes']==b[0]['routes']
 for x,y in zip(a[0]['memory'],b[0]['memory']):
  for k in ['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k],(ctx,pool,k)
 if '0'in results:
  old=json.load(open(O/'metadata-initial/metadata-100352-0.json'))
  # The old pool1 OFF metadata is not a pool0 baseline; only its model/KV routes should agree.
  assert results['0'][0]['routes']==old['routes']
 j.setdefault('pairs',[]).append({'context':ctx,'pool':pool,'default_equals_explicit1':True,'zero_has_local_nodes':False if '0'in results else None})
j['complete']=True;j['source_file_sha256']=hashlib.sha256((R/'src/llama-context.cpp').read_bytes()).hexdigest();(P/'results.json').write_text(json.dumps(j,indent=2)+'\n')
s=(R/'src/llama-context.cpp').read_text();old=(P/'BEFORE.cpp').read_text();(P/'promotion.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='before/src/llama-context.cpp',tofile='after/src/llama-context.cpp')));print('DEFAULT_PROMOTION_PASS',flush=True)
