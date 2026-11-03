from pathlib import Path
from urllib.request import Request,urlopen
import os,json,subprocess,time,signal,hashlib,re,threading,socket,statistics
from memory_monitor import MemoryMonitor
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp')
cfg=json.load(open(P/'config.json'));assert json.load(open(P/'precision-results.json'))['complete']
cmd=cfg['command'][:];cmd[cmd.index('--port')+1]='19087';cmd[cmd.index('--alias')+1]='pool0-default-screen'
base_env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};base_env.update(cfg['environment'])
tokens=json.load(open('/tmp/dsv41-m2-residual-profile-20261103/inputs.json'))['tokens']
plan=[(512,'warmup',-1)]+[(512,'measured',i)for i in range(2)]+[(8192,'warmup',-1)]+[(8192,'measured',i)for i in range(3)]+[(98304,'warmup',-1),(98304,'measured',0)]
j={'complete':False,'scope':'Clean native candidate-block OFF/ON/OFF; actual pooling defaultOFF/unset in all arms; block OFF/default/OFF, HC sharing stays defaultON,one process per arm,full uncached512/8K/98K,TG256,Q3/MXFP8Engram/Q8KV/BF16K3/full/pmin0.5/defaultM2/LOCAL/ASYNCoff,ctx100352/B8192/U512/P1/TopK2048,frozen placement/cache/staging/mmproj0.512warm+2,8Kwarm+3,98Kwarm+1 measured per arm;98Ksingleton and one arm order limit inference.No profiling/callbacks/interposers,calibration or builds during timed runs.','head':cfg['source_head'],'runtime_patch':cfg['runtime_patch'],'library_hashes':cfg['source_library_hashes'],'arms':[]}
def save():(P/'bench-results.json').write_text(json.dumps(j,indent=2)+'\n')
def api(path,body=None,timeout=5):
 with urlopen(Request('http://127.0.0.1:19087'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=timeout)as f:return json.load(f)
for label,value in [('off-pre','0'),('on',None),('off-post','0')]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
 for n,h in cfg['source_library_hashes'].items():assert hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()==h
 e=base_env.copy();e.pop('LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL',None)
 if value is not None:e['LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL']=value
 assert 'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in e and 'LD_PRELOAD'not in e
 d=P/('bench-'+label);d.mkdir(exist_ok=False)
 a={'label':label,'complete':False,'command':cmd,'environment':{k:v for k,v in e.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'requests':[]};j['arms'].append(a);save()
 proc=None;mon=None;stop=threading.Event();watch=None;floor=[]
 try:
  with (d/'server.log').open('wb')as log:
   started=time.monotonic();proc=subprocess.Popen(cmd,cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['pid']=proc.pid;mon=MemoryMonitor(proc.pid)
   def guard():
    while not stop.wait(.2):
     with mon.lock:
      breach=any(v.get('min_free',512*1048576)<512*1048576 for phase in ['load','inference']for v in mon.stats[phase].values()if isinstance(v,dict))
     if breach:
      floor.append(time.monotonic())
      if proc.poll()is None:proc.send_signal(signal.SIGTERM)
      return
   watch=threading.Thread(target=guard,daemon=True);watch.start()
   while True:
    assert proc.poll()is None,'load exit '+str(proc.returncode)
    try:
     if api('/health').get('status')=='ok':break
    except OSError:pass
    assert time.monotonic()-started<1200,'startup timeout';time.sleep(1)
   a['load_seconds']=time.monotonic()-started;mon.inference();save();print('BENCH_READY',label,flush=True)
   for n,kind,rep in plan:
    body={'prompt':tokens[:n],'n_predict':256,'temperature':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'seed':1234,'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'stream':False,'id_slot':0}
    v=api('/completion',body,1200);t=v['timings'];assert t['prompt_n']==n and t['cache_n']==0 and t['predicted_n']==len(v['tokens'])==256
    a['requests'].append({'n':n,'kind':kind,'rep':rep,'response':v,'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest()});save();assert not floor
    print('CASE',label,n,kind,rep,json.dumps(t),flush=True)
  a['complete']=True
 except Exception as ex:a['error']=repr(ex);raise
 finally:
  stop.set()
  if watch:watch.join(timeout=5)
  if proc and proc.poll()is None:
   proc.send_signal(signal.SIGTERM)
   try:proc.wait(timeout=60)
   except subprocess.TimeoutExpired:proc.kill();proc.wait();a['forced_kill']=True
  if mon:a['memory']=mon.finish()
  a['exit_code']=None if proc is None else proc.returncode;a['floor_hit']=floor
  log=(d/'server.log').read_text(errors='replace');a['warnings']=[l for l in log.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)]
  a['m2_active']='prefill_skip:'in log;a['local_active']='producer-local features enabled'in log;save()
 assert a['complete']and a['exit_code']==0 and not a['warnings']and not a.get('forced_kill')and not floor
 assert a['m2_active']and a['local_active']
 assert not a['memory'].get('error')
 assert all(v['min_free']>=512*1048576 for phase in ['load','inference']for v in a['memory'][phase].values()if isinstance(v,dict)and'min_free'in v)
 print('BENCH_ARM_DONE',label,flush=True)
for n in [512,8192,98304]:
 q=[x for a in j['arms']for x in a['requests']if x['n']==n]
 assert len({x['tokens_sha256']for x in q})==1,('token drift',n)
 assert len({tuple(x['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])for x in q})==1,('draft drift',n)
j['summary']=[]
for n in [512,8192,98304]:
 row={'n':n,'arms':{}}
 for a in j['arms']:
  q=[x['response']['timings']for x in a['requests']if x['n']==n and x['kind']=='measured']
  row['arms'][a['label']]={'repeats':len(q),'PP':statistics.median(x['prompt_per_second']for x in q),'TG':statistics.median(x['predicted_per_second']for x in q),'PP_range':[min(x['prompt_per_second']for x in q),max(x['prompt_per_second']for x in q)],'TG_range':[min(x['predicted_per_second']for x in q),max(x['predicted_per_second']for x in q)]}
 for metric in ['PP','TG']:
  ref=statistics.mean(row['arms'][a][metric]for a in ['off-pre','off-post'])
  row[metric+'_delta_pct']=100*(row['arms']['on'][metric]/ref-1)
  row[metric+'_off_drift_pct']=100*(row['arms']['off-post'][metric]/row['arms']['off-pre'][metric]-1)
 j['summary'].append(row)
j['complete']=True;save();print('BENCH_COMPLETE',json.dumps(j['summary']),flush=True)
