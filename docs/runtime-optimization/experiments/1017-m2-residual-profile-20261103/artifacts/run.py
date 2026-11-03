from pathlib import Path
from urllib.request import Request,urlopen
import os,json,hashlib,subprocess,time,signal,socket,re,threading,statistics
from memory_monitor import MemoryMonitor
P=Path(__file__).resolve().parent;R=Path('/home/sergei/Github/llama.cpp');NSYS='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'
old=json.load(open('/tmp/dsv41-dspark-tail-screen-20261103/bench-results.json'));base=old['arms']['on']['command'][:];base[base.index('--port')+1]='19059';base[base.index('--alias')+1]='m2-residual-profile'
env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(old['arms']['on']['environment']);env.pop('LLAMA_DSV41_PREFILL_SKIP',None)
ids=json.load(open('/tmp/dsv41-dspark-tail-screen-20261103/inputs.json'))['cases'][0]['tokens'];assert len(ids)>=98304
(P/'inputs.json').write_text(json.dumps({'tokens':ids[:98304]},separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(e):return{k:v for k,v in e.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))}
report={'complete':False,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'patch':subprocess.check_output(['git','diff'],cwd=R,text=True),'scope':'M2 default ON residual profile. Same Q3/MXFP8 Engram/Q8 KV/BF16K3/full/pmin0.5,ctx100352/B8192/U512/P1/TopK2048/frozen allGPU. Native warm+1 control and separate NSYS/interposer warm/pre/profile/post. 16 fixed greedy outputs for profiling, not a new TG benchmark. No eval/abort callbacks, source edits or default changes. Full uncached512/8K/98K. NSYS trace includes kernels after HTTP response only in labeled drain. floor512 perGPU.','binary_sha256':{n:sha(R/'build-aliceai/bin'/n)for n in ['llama-server','libllama.so','libllama-common.so','libggml-base.so','libggml-cpu.so','libggml-cuda.so']},'nsys_version':subprocess.check_output([NSYS,'--version'],text=True).strip(),'shim_sha256':sha(P/'shim.so'),'inputs_sha256':sha(P/'inputs.json'),'base_environment':safe(env),'variants':[]}
def save():(P/'results.json').write_text(json.dumps(report,indent=2)+'\n')
def atomic(p,s):q=p.with_suffix('.tmp');q.write_text(s);q.replace(p)
def api(path,body=None,timeout=5):
 with urlopen(Request('http://127.0.0.1:19059'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=timeout)as f:return json.load(f)
assert not report['patch'] and not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True)
save()
for variant in ['native','sidecar']:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 for n,h in report['binary_sha256'].items():assert sha(R/'build-aliceai/bin'/n)==h
 with socket.socket()as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',19059))
 root=P/variant;root.mkdir(exist_ok=True);assert not(root/'server.log').exists(),'refusing overwrite'
 e=env.copy();profiled=variant=='sidecar'
 if profiled:e.update(DSV_PROFILE_ARM=str(root/'arm'),DSV_PROFILE_STOP=str(root/'stop'),DSV_PROFILE_ACK=str(root/'ack'),DSV_PROFILE_PID=str(root/'pid'),DSV_PROFILE_EVENTS=str(root/'events.jsonl'))
 command=base[:]
 if profiled:command=[NSYS,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=repeat:3','--cuda-graph-trace=node','--force-overwrite=true','--output='+str(root/'capture'),'env','LD_PRELOAD='+str(P/'shim.so'),*base]
 row={'variant':variant,'command':base,'launch_command':command,'environment':safe(e),'complete':False,'requests':[]};report['variants'].append(row);save();proc=None;pid=None;mon=None;watch=None;stop=threading.Event();floor_hit=[];case_id=0
 def watch_floor():
  while not stop.wait(.2):
   try:
    if min(x['free']for x in mon.sample())<512*1048576:
     floor_hit.append(time.monotonic());os.kill(pid,signal.SIGTERM);return
   except ProcessLookupError:return
   except Exception as ex:floor_hit.append(str(ex));os.kill(pid,signal.SIGTERM);return
 def request(n,kind,trace=False):
  global case_id
  case_id+=1
  if profiled:atomic(root/'arm',f'{case_id} {int(trace)} {n}\n')
  print('REQUEST',variant,n,kind,flush=True);begin=time.monotonic();v=api('/completion',{'prompt':ids[:n],'n_predict':16,'temperature':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'seed':1234,'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'stream':False,'id_slot':0},2400);t=v['timings'];assert t['cache_n']==0 and t['prompt_n']==n and t['predicted_n']==len(v['tokens'])==16,t
  item={'id':case_id,'n':n,'kind':kind,'trace':trace,'wall_seconds':time.monotonic()-begin,'response':v,'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest()}
  if profiled:
   atomic(root/'stop',str(case_id)+'\n');limit=time.monotonic()+120
   while not(root/'ack').exists()or(root/'ack').read_text().strip()!=str(case_id):
    assert proc.poll()is None,'profiler exited';assert time.monotonic()<limit,'stop ack timeout';time.sleep(.05)
   events=[json.loads(x)for x in(root/'events.jsonl').read_text().splitlines()];cur=[x for x in events if x.get('case')==case_id];ubs=[x for x in cur if x.get('event')=='ubatch'and x['role']=='target_prefill'];assert sum(x['tokens']for x in ubs)==n,('observer ubatch mismatch',n,sum(x['tokens']for x in ubs));assert all(x['tokens']<=512 for x in ubs)
   item['target_ubatches']=len(ubs);item['kv_only_ubatches']=sum(x.get('event')=='m2_kv_only'for x in cur);item['draft_seed_rows']=sum(x['tokens']for x in cur if x.get('event')=='ubatch'and x['role']=='draft_seed')
   assert item['draft_seed_rows']>0,'draft observer missing'
   if n>=8192:assert item['kv_only_ubatches']>0 and item['draft_seed_rows']<n,'M2 inactive or wrong role'
  row['requests'].append(item);save();assert not floor_hit,'memory watcher';print('DONE',variant,n,kind,json.dumps(t),flush=True);return v
 try:
  with(root/'server.log').open('wb')as log:
   started=time.monotonic();proc=subprocess.Popen(command,env=e,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   if not profiled:pid=proc.pid
   while True:
    assert proc.poll()is None,'startup exited'
    if pid is None and(root/'pid').exists():pid=int((root/'pid').read_text())
    if pid and mon is None:mon=MemoryMonitor(pid);watch=threading.Thread(target=watch_floor,daemon=True);watch.start()
    try:
     if api('/health').get('status')=='ok':break
    except OSError:pass
    assert time.monotonic()-started<1200,'startup timeout';time.sleep(1)
   row['server_pid']=pid;row['load_seconds']=time.monotonic()-started;row['props']=api('/props');assert row['props']['default_generation_settings']['n_ctx']==100352;mon.inference();save();print('READY',variant,flush=True)
   if profiled:
    a=request(512,'smoke-control');b=request(512,'smoke-profile',True);assert a['tokens']==b['tokens']
    for n in [8192,98304]:
     request(n,'warmup');a=request(n,'pre-control');b=request(n,'profile',True);c=request(n,'post-control');assert a['tokens']==b['tokens']==c['tokens']
   else:
    for n in [512,8192,98304]:
     request(n,'warmup');request(n,'native-control')
   os.kill(pid,signal.SIGTERM);proc.wait(timeout=180);assert proc.returncode==0
 finally:
  if proc and proc.poll()is None:
   if pid:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
   try:proc.wait(timeout=60)
   except subprocess.TimeoutExpired:proc.kill();proc.wait();row['forced_kill']=True
  stop.set()
  if watch:watch.join(timeout=5)
  if mon:row['memory']=mon.finish()
  row['exit_code']=None if proc is None else proc.returncode;row['floor_hit']=floor_hit
  text=(root/'server.log').read_text(errors='replace');row['warnings']=[l for l in text.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error|\bE ',l,re.I)];row['skip_lines']=[l for l in text.splitlines()if'prefill_skip:'in l];row['seed_lines']=[l for l in text.splitlines()if'dspark_prefill_tail:'in l];save()
 assert row['exit_code']==0 and not row['warnings'] and not row.get('forced_kill') and not floor_hit,row['warnings']
 assert row['skip_lines'] and row['seed_lines']
 if profiled:
  for n in [512,8192,98304]:
   groups=[q for v in report['variants']for q in v['requests']if q['n']==n];assert len({q['tokens_sha256']for q in groups})==1,'observer output drift'
   assert len({tuple(q['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])for q in groups})==1,'draft counter drift'
  row['reports']=[]
  for f in sorted(root.glob('*.nsys-rep')):
   sql=f.with_suffix('.sqlite')
   with(root/(f.stem+'-export.log')).open('wb')as out:rc=subprocess.run([NSYS,'export','--type=sqlite','--force-overwrite=true','--output='+str(sql),str(f)],stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0;row['reports'].append({'report':str(f),'sqlite':str(sql)})
  assert len(row['reports'])==3,row['reports']
 row['complete']=True;save();print('VARIANT_DONE',variant,flush=True)
report['complete']=True;save();print('PROFILE_DONE',flush=True)
