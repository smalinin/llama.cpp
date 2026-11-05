from pathlib import Path
from urllib.request import Request,urlopen
import json,sys,os,hashlib,subprocess,time,threading,signal,re,traceback,shutil
R=Path('/home/sergei/Github/llama.cpp');B=R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts';P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts/copies8'
sys.path.insert(0,str(P))
from memory_monitor import MemoryMonitor

def save(f,j):q=f.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(f)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(pid):
 x=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return{'cpu_s':(int(x[11])+int(x[12]))/os.sysconf('SC_CLK_TCK'),'major_faults':int(x[9]),'minor_faults':int(x[7])}
def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not(P/'counts-results.json').exists()
 cfg=json.loads((B/'config.json').read_text());ids=json.loads((B/'inputs.json').read_text())['tokens']
 cfg['previous_library_hashes']=cfg['source_library_hashes'].copy();cfg['source_library_hashes']={f:sha(f)for f in cfg['source_library_hashes']};cfg['source_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();cfg['source_files']['ggml/src/ggml-backend.cpp']=sha(R/'ggml/src/ggml-backend.cpp')
 env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LD_PRELOAD=str(P/'probe.so'),DSV_RESIDUAL_ARM=str(P/'probe-arm'),DSV_RESIDUAL_LOG=str(P/'scopes.jsonl'),LD_LIBRARY_PATH=str(P/'candidate')+':'+str(R/'build-aliceai/bin'));env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None);env['CUDA_SCALE_LAUNCH_QUEUES']='4x'
 cmd=cfg['command'][:];cmd[cmd.index('--port')+1]='19097';cmd[cmd.index('--alias')+1]='prefill-copies8'
 j={'complete':False,'kind':'Copies8 screen, same phase3 probe and queue4x fixed; only existing compile-time GGML_SCHED_MAX_COPIES4->8 changes in a private base library. Host scopes plus CUDA13 actual launch ABI and broader CUDA/driver/CUBLAS API counters inside GPU graph_compute. No CUPTI/sampling/callbacks/extra GPU events/math/source/default changes. Pass-through controls before/after one armed8K request, same process. CPU lookup early defaultON/unset.','head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'config':cfg,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'probe_sha256':sha(P/'probe.so'),'requests':[]}
 persist=lambda:save(P/'counts-results.json',j);persist();proc=None;mon=None;halt=threading.Event();floor=[];guard=None
 try:
  with(P/'counts-server.log').open('wb')as log:
   t=time.monotonic();proc=subprocess.Popen(cmd,env=env,cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);j['pid']=proc.pid;persist();mon=MemoryMonitor(proc.pid)
   def watch():
    while not halt.wait(.2):
     with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for ph in ['load','inference']for x in mon.stats[ph].values()if isinstance(x,dict))
     if bad:
      floor.append(time.time())
      if proc.poll()is None:proc.send_signal(signal.SIGTERM)
      return
   guard=threading.Thread(target=watch,daemon=True);guard.start()
   def api(body=None):
    with urlopen(Request('http://127.0.0.1:19097'+('/completion'if body else'/health'),data=json.dumps(body).encode()if body else None,headers={'Content-Type':'application/json'}),timeout=1500 if body else 3)as f:return json.load(f)
   while True:
    assert proc.poll()is None
    try:
     if api().get('status')=='ok':break
    except OSError:pass
    assert time.monotonic()-t<1500;time.sleep(1)
   j['ready_seconds']=time.monotonic()-t;j['loaded_base_maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if 'libggml-base.so'in x];j['private_base_sha256']=sha(P/'candidate/libggml-base.so.0');assert any(str(P/'candidate')in x for x in j['loaded_base_maps']);persist();mon.inference();print('READY',flush=True)
   def request(toks,n):return api(dict(prompt=toks,n_predict=n,temperature=0,top_k=1,top_p=1,min_p=0,repeat_penalty=1,seed=1234,ignore_eos=True,cache_prompt=False,return_tokens=True,id_slot=0))
   case=0
   for n in [8192]:
    for kind in (['warmup','warmup','pre','counts','post']if n==8192 else['warmup','pre','counts','post']):
     request([97],1);case+=1
     if kind=='counts':(P/'probe-arm').write_text(str(case)+'\n')
     a=stats(proc.pid);start=time.monotonic();v=request(ids[:n],16);wall=time.monotonic()-start;z=stats(proc.pid)
     if(P/'probe-arm').exists():(P/'probe-arm').unlink()
     t=v['timings'];assert t['cache_n']==0 and t['prompt_n']==n and t['predicted_n']==len(v['tokens'])==16
     row={'case':case,'prefix':n,'kind':kind,'response':v,'wall_s':wall,'before':a,'after':z,'pp':n*1000/t['prompt_ms'],'tg':15000/t['predicted_ms'],'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest()};j['requests'].append(row);persist();assert not floor
     print('CASE',case,n,kind,round(row['pp'],2),round(row['tg'],2),flush=True)
   j['requests_complete']=True
 except Exception as ex:j['error']=repr(ex);j['traceback']=traceback.format_exc();raise
 finally:
  if proc and proc.poll()is None:
   proc.send_signal(signal.SIGTERM)
   try:proc.wait(timeout=60)
   except subprocess.TimeoutExpired:proc.kill();proc.wait();j['forced_kill']=True
  halt.set()
  if guard:guard.join(timeout=3)
  if mon:j['memory']=mon.finish()
  j['exit_code']=proc.returncode if proc else None;j['floor_breach']=floor;j['warnings']=[l for l in(P/'counts-server.log').read_text(errors='replace').splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)];persist()
 assert j.get('requests_complete')and j['exit_code']==0 and not j.get('forced_kill')and not floor and not j['warnings']
 for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
 ref={}
 for row in j['requests']:
  val=(row['tokens_sha256'],row['response']['timings'].get('draft_n'),row['response']['timings'].get('draft_n_accepted'));key=row['prefix']
  if key not in ref:ref[key]=val
  assert ref[key]==val,('output/counters mismatch',row['case'])
 assert 'copies=8'in(P/'counts-server.log').read_text(errors='replace');j['copies8_activation_confirmed']=True;j['all_outputs_counters_exact']=True;j['complete']=True;persist();print('RESIDUAL_COMPLETE',flush=True)
if __name__=='__main__':main()
