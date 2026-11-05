from pathlib import Path
from urllib.request import Request,urlopen
import os,sys,json,time,subprocess,signal,threading,hashlib,re,traceback,socket
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent;assert P.parent.name=='1017-verify-float6-acceptance-20261106'
B=R/'docs/runtime-optimization/experiments/1017-verify-float6-20261106/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save(f,j):q=f.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(f)
def main():
 gate=json.loads((P/'lifecycle-results.json').read_text());assert gate['complete']and gate['all_groups_exact']and gate['dispatch_reduction_verified'];cfg=json.loads((P/'config.json').read_text());ids=json.loads((P/'native-inputs.json').read_text())['tokens'];candidate=B/'candidate/libllama.so.0';runtime=json.loads((P/'runtime-build.json').read_text());assert sha(candidate)==runtime['private_llama_sha256'];assert not(P/'native-results.json').exists()
 port=19108
 with socket.socket()as s:s.bind(('127.0.0.1',port))
 j=dict(complete=False,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),config=cfg,private_library_sha256=sha(candidate),native_input_sha256=sha(P/'native-inputs.json'),port=port,arms=[],protocol='Same private3bb module/flags FLOAT_WIDE0-1-0, MoE_WIDE1/prefilldeferral0. Clean native, no interposer/CUPTI/callback/diagnostic timing. Q3/Q8KV/TopK2048/K5/full/pmin.5/ctx92160/B8192/U512/P1/threads12/new-hostUUIDs/frozen placement. 512/8K/88K one warmup+two measured requests each, reset before every request, full cache_n0;256greedy outputs/255TG intervals. Reject any output-array or draft-counter mismatch. Floor512MiB.')
 persist=lambda:save(P/'native-results.json',j);persist();references={}
 for label,flag in [('off-pre','0'),('on','1'),('off-post','0')]:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[x['uuid']for x in cfg['new_gpu_map']]
  for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
  for f,h in cfg['source_files'].items():assert sha(R/f)==h,f
  assert sha(candidate)==j['private_library_sha256']
  for m in cfg['model_stats']:
   st=Path(m['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(m[k]for k in ['device','inode','bytes','mtime_ns'])
  env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LD_LIBRARY_PATH=str(candidate.parent)+':'+str(R/'build-aliceai/bin'),LLAMA_DSV41_VERIFY_MOE_WIDE='1',LLAMA_DSV41_VERIFY_FLOAT_WIDE=flag,LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSPARK_CUM_P_MIN='0',LLAMA_DSPARK_EMPTY_COOLDOWN='1');env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None)
  cmd=cfg['command'][:];cmd[cmd.index('--port')+1]=str(port);cmd[cmd.index('--host')+1]='127.0.0.1';cmd[cmd.index('--alias')+1]='float6-'+label;cmd[cmd.index('--spec-draft-p-min')+1]='0.5';cmd[cmd.index('--spec-draft-n-max')+1]='5';d=P/('native-'+label);d.mkdir()
  a=dict(label=label,flag=flag,complete=False,command=cmd,environment={k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','LD_'))},requests=[]);j['arms'].append(a);persist();proc=None;mon=None;halt=threading.Event();floor=[];watcher=None
  try:
   with(d/'server.log').open('wb')as log:
    start=time.monotonic();proc=subprocess.Popen(cmd,cwd=d,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['pid']=proc.pid;persist();mon=MemoryMonitor(proc.pid)
    def watch():
     while not halt.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for phase in ['load','inference']for x in mon.stats[phase].values()if isinstance(x,dict))
      if bad:
       floor.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    def api(body=None):
     with urlopen(Request('http://127.0.0.1:'+str(port)+('/completion'if body else'/health'),data=json.dumps(body).encode()if body else None,headers={'Content-Type':'application/json'}),timeout=1500 if body else 3)as f:return json.load(f)
    while True:
     assert proc.poll()is None
     try:
      if api().get('status')=='ok':break
     except OSError:pass
     assert time.monotonic()-start<1800;time.sleep(1)
    a['ready_seconds']=time.monotonic()-start;a['maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert a['maps']and all(str(candidate.parent)in x for x in a['maps']);persist();print('READY_NATIVE',label,flush=True);mon.inference()
    def request(tokens,n):return api(dict(prompt=tokens,n_predict=n,temperature=0,top_k=1,top_p=1,min_p=0,repeat_penalty=1,seed=1234,ignore_eos=True,cache_prompt=False,return_tokens=True,id_slot=0))
    for n in [512,8192,90112]:
     for kind in ['warmup','measured','measured']:
      request([97],1);t0=time.monotonic();v=request(ids[:n],256);wall=time.monotonic()-t0;t=v['timings'];assert t['cache_n']==0 and t['prompt_n']==n and t['predicted_n']==len(v['tokens'])==256
      row=dict(prefix=n,kind=kind,pp=n*1000/t['prompt_ms'],tg=255000/t['predicted_ms'],tokens_sha256=hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest(),response=v,wall_s=wall);a['requests'].append(row);persist();print('CASE_NATIVE',label,n,kind,round(row['pp'],2),round(row['tg'],2),flush=True)
      control=(v['tokens'],t['draft_n'],t['draft_n_accepted'])
      if n in references:assert references[n]==control,('native output/draft counter mismatch',label,n,kind)
      else:references[n]=control
      assert not floor
    a['complete']=True
  except Exception as ex:a['error']=repr(ex);a['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();a['forced_kill']=True
   halt.set()
   if watcher:watcher.join(timeout=3)
   if mon:a['memory']=mon.finish()
   a['floor_breach']=floor;a['exit_code']=proc.returncode if proc else None;txt=(d/'server.log').read_text(errors='replace');a['warnings']=[x for x in txt.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',x,re.I)];persist()
  assert a['complete']and a['exit_code']==0 and not a.get('forced_kill')and not floor and not a['warnings'];print('NATIVE_ARM_DONE',label,flush=True)
 for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
 j.update(complete=True,all_outputs_counters_exact=True);persist();print('NATIVE_COMPLETE',flush=True)
if __name__=='__main__':main()
