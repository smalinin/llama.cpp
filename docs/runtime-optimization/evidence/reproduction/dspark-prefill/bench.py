from pathlib import Path
from urllib.request import Request,urlopen
import os,json,subprocess,time,hashlib,signal,socket,re,statistics
from memory_monitor import MemoryMonitor
P=Path(__file__).resolve().parent;R=Path('/home/sergei/Github/llama.cpp')
assert json.load(open(P/'long-native/results.json'))['complete'],'long correctness gate incomplete'
assert json.load(open(P/'transport-native/results.json'))['complete'],'transport gate incomplete'
base=json.load(open(P/'long-manifest.json'));args=base['command'][3:];args.remove('--no-spec-draft-backend-sampling');args[args.index('--spec-draft-n-max')+1]='3';args[args.index('--spec-draft-p-min')+1]='0.5';env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV41_'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(base['environment']);env.pop('LD_PRELOAD',None)
for k in list(env):
 if k.startswith('DSV41_'):env.pop(k)
tokens=json.load(open(P/'inputs.json'))['cases'][0]['tokens'];plan=[(n,w,r)for n in [8192,98304]for w,r in [(True,-1),(False,0),(False,1),(False,2)]]
j={'complete':False,'scope':'Q3/MXFP8 Engram/Q8KV + BF16 DSpark K3/full/p-min0.5/backend-samplingON/default LOCAL/ASYNCoff. ctx100352/B8192/U512/P1/TopK2048/frozenallGPU/fitOFF/cache-staging0/mmprojOFF. Full uncached8K/98K warm+3,256greedy. Single OFF->ON, no profiler/interposer. Not a Maya comparison.','head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'patch':subprocess.check_output(['git','diff'],cwd=R,text=True),'hashes':{n:hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()for n in ['llama-server','libllama.so','libllama-common.so','libggml-cuda.so']},'arms':{}}
def save():(P/'bench-results.json').write_text(json.dumps(j,indent=2)+'\n')
def api(path,body=None):
 with urlopen(Request('http://127.0.0.1:19049'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=600 if body else 5)as f:return json.load(f)
save()
try:
 for mode in ['off','on']:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
  for n,h in j['hashes'].items():assert hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()==h
  env.pop('LLAMA_DSV41_PREFILL_SKIP',None)
  if mode=='on':env['LLAMA_DSV41_PREFILL_SKIP']='1'
  cmd=[str(R/'build-aliceai/bin/llama-server')]+args+['--host','127.0.0.1','--port','19049','--alias','prefill-skip-dspark']
  with socket.socket()as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',19049))
  d=P/('bench-'+mode);d.mkdir(exist_ok=True);a={'complete':False,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV41_','LD_'))},'requests':[]};j['arms'][mode]=a;save();proc=None;mon=None
  try:
   with (d/'server.log').open('wb')as log:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['pid']=proc.pid;mon=MemoryMonitor(proc.pid);started=time.monotonic();save();print('START',mode,flush=True)
    while True:
     assert proc.poll()is None,'startup exit'
     try:
      if api('/health').get('status')=='ok':break
     except OSError:pass
     assert time.monotonic()-started<900,'load timeout';time.sleep(1)
    a['load_seconds']=time.monotonic()-started;mon.inference();print('READY',mode,flush=True)
    for n,w,r in plan:
     body={'prompt':tokens[:n],'n_predict':256,'temperature':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'seed':1234,'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'stream':False,'id_slot':0}
     v=api('/completion',body);ts=v['timings'];assert ts['prompt_n']==n and ts['cache_n']==0 and len(v['tokens'])==ts['predicted_n']==256,ts
     a['requests'].append({'n':n,'warmup':w,'rep':r,'response':v,'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest()});save();print('CASE',mode,n,r,json.dumps(ts),flush=True)
     assert min(x['free']for x in mon.sample())>=512*1048576,'free VRAM below512MiB'
    a['complete']=True
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();a['forced_kill']=True
   if mon:a['memory']=mon.finish()
   a['exit_code']=None if proc is None else proc.returncode
   text=(d/'server.log').read_text(errors='replace');a['warnings']=[l for l in text.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)];a['skip_lines']=[l for l in text.splitlines()if'prefill_skip:'in l];a['seed_lines']=[l for l in text.splitlines()if'dspark_prefill_tail:'in l];save()
  assert a['exit_code']==0 and not a['warnings'] and not a.get('forced_kill'),a.get('warnings')
  if mode=='on':
   assert sum('tokens=8192 cut=4608 keep=3584' in l for l in a['skip_lines'])==44,a['skip_lines']
   assert sum('tokens=7676 cut=4096 keep=3580' in l for l in a['skip_lines'])==8,a['skip_lines']
   assert len(a['skip_lines'])==52,a['skip_lines']
   assert len(a['seed_lines'])==52,a['seed_lines']
   assert sum('tokens=8192 advanced=7168 injected=1024'in l for l in a['seed_lines'])==44
   assert sum('tokens=7676 advanced=6656 injected=1020'in l for l in a['seed_lines'])==8
  else:assert not a['skip_lines'] and not a['seed_lines']
  assert all(x['min_free']>=512*1048576 for x in a['memory']['inference'].values()if isinstance(x,dict)and 'min_free'in x)
 for n in [8192,98304]:assert len({r['tokens_sha256']for a in j['arms'].values()for r in a['requests']if r['n']==n})==1,'paired output drift'
 for n in [8192,98304]:assert len({tuple(q['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])for a in j['arms'].values()for q in a['requests']if q['n']==n})==1,'draft counters drift'
 j['summary']=[]
 for n in [8192,98304]:
  row={'n':n}
  for mode,a in j['arms'].items():
   qs=[q['response']['timings']for q in a['requests']if q['n']==n and not q['warmup']]
   row[mode]={k:statistics.median(q[k]for q in qs)for k in ['prompt_per_second','predicted_per_second']}
  row['pp_change_pct']=100*(row['on']['prompt_per_second']/row['off']['prompt_per_second']-1);row['tg_change_pct']=100*(row['on']['predicted_per_second']/row['off']['predicted_per_second']-1);j['summary'].append(row)
 j['complete']=True;save();print('BENCH_DONE',json.dumps(j['summary']),flush=True)
except Exception as e:j['error']=repr(e);save();raise
