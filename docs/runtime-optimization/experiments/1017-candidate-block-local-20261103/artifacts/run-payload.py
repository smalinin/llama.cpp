from pathlib import Path
import os,json,subprocess,time,signal,hashlib,re,threading
from urllib.request import Request,urlopen
from memory_monitor import MemoryMonitor
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');bench=json.load(open(P/'config.json'));admit=json.load(open(P/'payload-admission.json'));assert admit['complete'];ctx=admit['chosen_context'];proposal=json.load(open(P/f'metadata-{ctx}-1.json'));assert proposal['fits_512mib'];root=P/f'payload-{ctx}';root.mkdir(exist_ok=False);cmd=bench['command'][:];cmd[cmd.index('--ctx-size')+1]=str(ctx);cmd[cmd.index('--port')+1]='19078';cmd[cmd.index('--alias')+1]='candidate-block-large-context';env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV41_','DSV_'))and k not in['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(bench['environment']);env.update(LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL='1',LD_PRELOAD=str(P/'placement.so')+':'+str(P/'transfer-audit.so'),DSV_PLACEMENT_FILE=str(root/'placement.jsonl'),DSV_TRANSFER_FILE=str(root/'audit.jsonl'),DSV_TRANSFER_ARM=str(root/'arm'))
for n in ['GGML_SCHED_COPY_RESHAPE','LLAMA_DSV41_PREFILL_SKIP','LLAMA_DSPARK_GPU_FEATURES','LLAMA_DSPARK_GPU_FEATURES_LOCAL','LLAMA_DSPARK_GPU_FEATURES_ASYNC']:assert n not in env
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
base=json.load(open('/tmp/dsv41-m2-residual-profile-20261103/inputs.json'))['tokens'];near=ctx-4096;tokens=(base*((near+len(base)-1)//len(base)))[:near];(root/'inputs.json').write_text(json.dumps({'tokens':tokens,'near':near,'source':'repeat frozen98304 IDs; lifecycle/memory stress, not task quality'},separators=(',',':'))+'\n')
j={'complete':False,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'scope':'Actual larger-context candidate-block ON/poolON/HCsharingON/M2/LOCAL lifecycle/memory and read-only pointer-copy gate.8K+256,near-context+16,then8K+16 full uncached.Not speed A/B.Loader placement observer and external full-range metadata/CUDA API observer armed only for the near-context request.No profiler/eval callback/GPU buffer additions/payload reads/extra sync.','context':ctx,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV41_','DSV_'))or k in['LD_LIBRARY_PATH','LD_PRELOAD']},'proposal':str(P/f'metadata-{ctx}-1.json'),'admission':admit,'binary_sha256':{n:hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()for n in bench['source_library_hashes']},'requests':[]};save=lambda:(root/'results.json').write_text(json.dumps(j,indent=2)+'\n')
def api(path,body=None,timeout=5):
 with urlopen(Request('http://127.0.0.1:19078'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=timeout)as f:return json.load(f)
assert j['binary_sha256']==bench['source_library_hashes'],'library changed'
proc=None;mon=None;stop=threading.Event();watch=None
try:
 with(root/'server.log').open('wb')as log:
  t=time.monotonic();proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);j['pid']=proc.pid;mon=MemoryMonitor(proc.pid);save()
  def guard():
   while not stop.wait(.2):
    with mon.lock:
     breach=any(v.get('min_free',512*1048576)<512*1048576 for phase in ['load','inference']for v in mon.stats.get(phase,{}).values()if isinstance(v,dict))
    if breach:
     j['floor_breach']=True
     if proc.poll()is None:proc.send_signal(signal.SIGTERM)
     return
  watch=threading.Thread(target=guard,daemon=True);watch.start()
  while True:
   assert proc.poll()is None,'startup exit '+str(proc.returncode)
   try:
    if api('/health').get('status')=='ok':break
   except OSError:pass
   assert time.monotonic()-t<1200,'startup timeout';time.sleep(1)
  j['load_seconds']=time.monotonic()-t;j['props']=api('/props');assert j['props']['default_generation_settings']['n_ctx']==ctx;mon.inference();j['ready_min_free_mib']=min(v['free']/1048576 for v in mon.sample());assert j['ready_min_free_mib']>=512;save();print('PAYLOAD_READY',ctx,j['ready_min_free_mib'],flush=True)
  observed=[json.loads(l)for l in(root/'placement.jsonl').read_text().splitlines()];actual=next(x for x in observed if x['arch']=='deepseek41');j['actual_placement']=actual
  for k in ['routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles']:assert actual[k]==proposal[k],('placement',k)
  assert actual['layers']==proposal['routes'],'per-tensor placement differs'
  for n,predict in [(8192,256),(near,16),(8192,16)]:
   assert min(v['free']for v in mon.sample())>=512*1048576;body={'prompt':tokens[:n],'n_predict':predict,'temperature':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'seed':1234,'ignore_eos':True,'cache_prompt':False,'stream':False,'return_tokens':True,'id_slot':0};t=time.monotonic();print('PAYLOAD_REQUEST',n,predict,flush=True);
   if n==near:(root/'arm').write_text('on\n')
   v=api('/completion',body,3600);ts=v['timings'];
   if(root/'arm').exists():(root/'arm').unlink()
   assert not v.get('error'),v;assert ts['prompt_n']==n and ts['cache_n']==0 and ts['predicted_n']==len(v['tokens'])==predict;assert ts.get('draft_n',0)>0;assert min(x['free']for x in mon.sample())>=512*1048576;j['requests'].append({'prompt_n':n,'n_predict':predict,'wall_seconds':time.monotonic()-t,'response':v,'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest()});save();print('PAYLOAD_CASE',n,json.dumps(ts),flush=True)
  old=json.load(open('/tmp/dsv41-dspark-tail-screen-20261103/bench-results.json'));a=next(x for x in old['arms']['on']['requests']if x['n']==8192 and not x['warmup']);j['same_8k_tokens_as_benchmark']=j['requests'][0]['tokens_sha256']==a['tokens_sha256'];j['repeat_tokens_equal_first16']=j['requests'][2]['response']['tokens']==j['requests'][0]['response']['tokens'][:16];old_large=json.load(open('/tmp/dsv41-m2-large-context-20261103/payload-176128/results.json'));j['same_172k_tokens_as_old_copyOFF']=None if ctx!=176128 else j['requests'][1]['response']['tokens']==old_large['requests'][1]['response']['tokens'];assert j['same_172k_tokens_as_old_copyOFF']is not False;assert j['same_8k_tokens_as_benchmark'];j['complete']=True
except Exception as e:j['error']=repr(e);raise
finally:
 stop.set()
 if watch:watch.join(timeout=5)
 if proc and proc.poll()is None:
  proc.send_signal(signal.SIGTERM)
  try:proc.wait(timeout=60)
  except subprocess.TimeoutExpired:proc.kill();proc.wait();j['forced_kill']=True
 if mon:j['memory']=mon.finish()
 j['exit_code']=None if proc is None else proc.returncode;log=(root/'server.log').read_text(errors='replace')if(root/'server.log').exists()else'';j['allocation_warnings']=[l for l in log.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)];j['local_active']='producer-local features enabled, packet CUDA0 60.00 MiB'in log;j['host_packing_fallback']='GPU feature packing fallback to host'in log;save()
assert j['complete']and j['exit_code']==0 and not j.get('floor_breach')and not j.get('forced_kill')and not j['allocation_warnings'];assert j['local_active']and not j['host_packing_fallback'];assert j['repeat_tokens_equal_first16'];assert not j['memory'].get('error');assert all(v['min_free']>=512*1048576 for phase in ['load','inference']for v in j['memory'][phase].values()if isinstance(v,dict)and'min_free'in v);print('PAYLOAD_DONE',ctx,flush=True)
