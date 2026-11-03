from pathlib import Path
import os,json,subprocess,time,hashlib,signal,re
from memory_monitor import MemoryMonitor
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');O=Path('/tmp/dsv41-candidate-pool-20261103');cfg=json.load(open(P/'config.json'))
a=cfg['command'][1:];args=[];i=0
while i<len(a):
 k=a[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 args.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:args.append(a[i+1]);i+=2
args+=['--no-spec-draft-backend-sampling'];args[args.index('--spec-draft-n-max')+1]='5';args[args.index('--spec-draft-p-min')+1]='0'
refs=json.load(open(O/'fresh-on-native/results.json'));assert refs['complete'];ref={(x['corpus'],x['key']):x['result'] for x in refs['cases']}
j={'complete':False,'scope':'Two fresh contexts OFF then ON, no mutable topology flag. Full two98K corpora/pooling env unset (OFF); block OFF vs actual unset/defaultON. Existing raw target/draft logits/taps/state/K3K5/append/restore/rollback/NLL128/fault+retry, compared to historical exact reference as well. Not a speed benchmark.','libraries':cfg['source_library_hashes'],'arms':[]};save=lambda:(P/'precision-results.json').write_text(json.dumps(j,indent=2)+'\n')
for flag in ['0','default']:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
 for n,h in cfg['source_library_hashes'].items():assert hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()==h
 env={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','DSV','CUDA_'))and k not in['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(cfg['environment']);env.update(LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=flag,LD_PRELOAD=str(P/'trace.so'),DSV41_TRACE_FILE=str(P/f'precision-{flag}-trace.json'))
 if flag=='default':env.pop('LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL',None)
 assert 'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in env
 root=P/f'precision-{flag}';cmd=[str(P/'fresh'),str(root),str(P/'inputs.json'),*args];arm={'flag':flag,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','DSV','CUDA_','LD_'))}};j['arms'].append(arm);save();proc=None;mon=None;t0=time.monotonic()
 try:
  with(P/f'precision-{flag}.log').open('wb')as log:
   proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);arm['pid']=proc.pid;mon=MemoryMonitor(proc.pid);ready=False
   while proc.poll()is None:
    text=(P/f'precision-{flag}.log').read_text(errors='replace')
    if not ready and '\nREADY\n'in text:ready=True;mon.inference();print('READY',flag,flush=True)
    with mon.lock:breach=any(v.get('min_free',512*1048576)<512*1048576 for ph in ['load','inference']for v in mon.stats[ph].values()if isinstance(v,dict))
    if breach:arm['floor_breach']=True;raise RuntimeError('512MiB floor breach')
    assert time.monotonic()-t0<3600,'timeout';time.sleep(.2)
   arm['exit_code']=proc.returncode;assert proc.returncode==0,'native exit'+str(proc.returncode)
  result=json.load(open(root/'results.json'));assert result['complete']and len(result['cases'])==12 and len(result['faults'])==2
  for x in result['cases']:assert x['result']==ref[(x['corpus'],x['key'])],('raw mismatch',flag,x['corpus'],x['key'])
  arm['groups_exact']=12;arm['faults']=result['faults'];arm['complete']=True
 except Exception as ex:arm['error']=repr(ex);raise
 finally:
  if proc and proc.poll()is None:
   proc.send_signal(signal.SIGTERM)
   try:proc.wait(timeout=60)
   except subprocess.TimeoutExpired:proc.kill();proc.wait();arm['forced_kill']=True
  if mon:arm['memory']=mon.finish()
  arm['exit_code']=None if proc is None else proc.returncode;arm['seconds']=time.monotonic()-t0;save()
 text=(P/f'precision-{flag}.log').read_text(errors='replace');arm['warnings']=[l for l in text.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)];assert not arm['warnings']and not arm.get('forced_kill');inside=False;bad=[];faults=0
 for l in text.splitlines():
  if l=='FAULT_BEGIN':assert not inside;inside=True;faults+=1
  elif l=='FAULT_END':assert inside;inside=False
  elif re.search(r'\bE ',l)and not inside:bad.append(l)
 assert not inside and faults==2 and not bad,bad
 trace=json.load(open(P/f'precision-{flag}-trace.json.block.json'));arm['activation']=trace
 assert trace['active_graphs'][1]>0 if flag=='default' else trace['active_graphs']==[0,0]
 pool=json.load(open(P/f'precision-{flag}-trace.json.pool.json'));assert pool['remote'][0]>0 and not any(pool['local']);arm['pooling_placement']=pool
 if flag=='default':assert trace['legacy_dense'][1]==0
 save();print('ARM_COMPLETE',flag,'exact12/fault2',arm['seconds'],trace,flush=True)
j['complete']=True;save();print('PRECISION_COMPLETE',flush=True)
