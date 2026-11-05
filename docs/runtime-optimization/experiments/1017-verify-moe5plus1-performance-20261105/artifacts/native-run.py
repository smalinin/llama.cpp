from pathlib import Path
from urllib.request import Request,urlopen
import os,sys,json,time,subprocess,signal,threading,hashlib,re,traceback
R=Path('/home/sergei/Github/llama.cpp');M=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';F=R/'docs/runtime-optimization/experiments/1017-full-precision-forensic-20261105/artifacts';P=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-performance-20261105/artifacts';B=R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(f,j):q=f.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(f)
def main():
 cfg=json.loads((B/'config.json').read_text());cfg['previous_library_hashes']=cfg['source_library_hashes'].copy();cfg['source_library_hashes']={f:sha(f)for f in cfg['source_library_hashes']};cfg['previous_source_files']=cfg['source_files'].copy();cfg['source_files']={f:sha(R/f)for f in cfg['source_files']};cfg['source_files'].update({f:sha(R/f)for f in ['src/models/deepseek41.cpp','src/llama-kv-cache-dsv4.cpp','src/llama-kv-cache-dsv4.h']});cfg['runtime_patch']=subprocess.check_output(['git','diff','--','src'],cwd=R,text=True);ids=json.loads((B/'inputs.json').read_text())['tokens'];assert not(P/'native-results.json').exists();candidate=sha(M/'candidate/libllama.so.0');check=json.loads((F/'precision-results.json').read_text());assert check['complete'] and check['all_groups_exact'];assert json.loads((P/'admission.json').read_text())['admit'];assert json.loads((P/'width-results.json').read_text())['complete'];verified=json.loads((P/'config.json').read_text());assert cfg['source_library_hashes']==verified['source_library_hashes'];assert all(cfg['source_files'][f]==h for f,h in verified['source_files'].items());cfg['command'][cfg['command'].index('--ctx-size')+1]='92160'
 j={'complete':False,'config':cfg,'candidate_sha256':candidate,'source_sha256':sha(R/'src/models/deepseek41.cpp'),'source_header_sha256':sha(R/'src/llama-kv-cache-dsv4.h'),'protocol':'Clean MoE5+1 OFF/ON/OFF using same isolated9c440a18 candidate library; prefill cache deferral0 in all arms; no interposer/CUPTI/callback/diagnostic timing. New-host frozen Q3/Q8KV/K5/pmin.5/full/B8192/U512/P1/TopK2048/ctx92160/threads12. 256greedy output tokens,255 TG intervals. 512/8K/88K warm+two measured each; same input IDs and uncached full prompt. floor512MiB.','arms':[]};persist=lambda:save(P/'native-results.json',j);persist();plan=[(n,k)for n in [512,8192,90112]for k in ['warmup','measured','measured']]
 for label,flag in [('off-pre','0'),('on','1'),('off-post','0')]:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[x['uuid']for x in cfg['new_gpu_map']];assert sha(M/'candidate/libllama.so.0')==candidate
  for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
  for m in cfg['model_stats']:
   st=Path(m['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(m['device'],m['inode'],m['bytes'],m['mtime_ns'])
  d=P/('native-'+label);d.mkdir();env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(cfg['environment']);env.update(LD_LIBRARY_PATH=str(M/'candidate')+':'+str(R/'build-aliceai/bin'),LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSV41_VERIFY_MOE_WIDE=flag);env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None);cmd=cfg['command'][:];cmd[cmd.index('--port')+1]='19106';cmd[cmd.index('--alias')+1]='moe5plus1-'+label
  a={'label':label,'flag':flag,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','LD_'))},'requests':[],'complete':False};j['arms'].append(a);persist();proc=None;mon=None;halt=threading.Event();floor=[];guard=None
  try:
   with(d/'server.log').open('wb')as log:
    start=time.monotonic();proc=subprocess.Popen(cmd,cwd=d,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['pid']=proc.pid;persist();mon=MemoryMonitor(proc.pid)
    def watch():
     while not halt.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for ph in ['load','inference']for x in mon.stats[ph].values()if isinstance(x,dict))
      if bad:
       floor.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    guard=threading.Thread(target=watch,daemon=True);guard.start()
    def api(body=None):
     with urlopen(Request('http://127.0.0.1:19106'+('/completion'if body else'/health'),data=json.dumps(body).encode()if body else None,headers={'Content-Type':'application/json'}),timeout=1500 if body else 3)as f:return json.load(f)
    while True:
     assert proc.poll()is None
     try:
      if api().get('status')=='ok':break
     except OSError:pass
     assert time.monotonic()-start<1500;time.sleep(1)
    a['ready_seconds']=time.monotonic()-start;a['maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert a['maps']and all(str(M/'candidate')in x for x in a['maps']);persist();print('READY',label,flush=True)
    mon.inference()
    def request(tokens,n):return api(dict(prompt=tokens,n_predict=n,temperature=0,top_k=1,top_p=1,min_p=0,repeat_penalty=1,seed=1234,ignore_eos=True,cache_prompt=False,return_tokens=True,id_slot=0))
    for n,kind in plan:
     request([97],1);start=time.monotonic();v=request(ids[:n],256);wall=time.monotonic()-start;t=v['timings'];assert t['cache_n']==0 and t['prompt_n']==n and t['predicted_n']==len(v['tokens'])==256
     a['requests'].append({'prefix':n,'kind':kind,'pp':n*1000/t['prompt_ms'],'tg':255000/t['predicted_ms'],'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest(),'response':v,'wall_s':wall});persist();assert not floor;print('CASE',label,n,kind,round(a['requests'][-1]['pp'],2),round(a['requests'][-1]['tg'],2),flush=True)
    a['complete']=True
  except Exception as ex:a['error']=repr(ex);a['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();a['forced_kill']=True
   halt.set()
   if guard:guard.join(timeout=3)
   if mon:a['memory']=mon.finish()
   a['floor_breach']=floor;a['exit_code']=proc.returncode if proc else None;lines=(d/'server.log').read_text(errors='replace').splitlines();a['warnings']=[x for x in lines if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',x,re.I)];a['activation']=[x for x in lines if 'sched_cpu_lookup:'in x];persist()
  assert a['complete']and a['exit_code']==0 and not a.get('forced_kill')and not floor and not a['warnings'];print('ARM_DONE',label,flush=True)
 for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
 # Require matching observed output paths and draft counters before equivalent speed comparison.
 refs={}
 for a in j['arms']:
  for row in a['requests']:
   key=row['prefix'];value=(row['tokens_sha256'],row['response']['timings'].get('draft_n'),row['response']['timings'].get('draft_n_accepted'))
   if key not in refs:refs[key]=value
   assert refs[key]==value,('native output or draft counter mismatch',a['label'],key)
 j['all_outputs_counters_exact']=True;j['complete']=True;persist();print('NATIVE_COMPLETE',flush=True)
if __name__=='__main__':main()
