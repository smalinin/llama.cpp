from pathlib import Path
import os,sys,json,hashlib,subprocess,time,threading,signal,traceback
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):q=p.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def main():
 orig=json.loads((P/'precision-results.json').read_text());cfg=orig['config'];cmd=orig['arms'][0]['command'][3:];assert cmd[0]=='--model',cmd[:3];build=json.loads((P/'build.json').read_text());j={'complete':False,'scope':'Fresh OFF baseline first8K only, three clear/reset repeats, original state/tap capture order and same fault interposer as failed raw runs, no fault armed. Six-query candidate is inactive. Diagnostic reproducibility check, not native speed or full-model acceptance.','candidate_sha256':sha(P/'candidate/libllama.so.0'),'harness_sha256':sha(P/'baseline-prefix-harness'),'input_sha256':sha(P/'precision-inputs.json'),'arms':[]};dest=P/'baseline-prefix-results.json';assert not dest.exists();save(dest,j)
 for flag in ['0']:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert sha(P/'candidate/libllama.so.0')==j['candidate_sha256'];assert all(sha(R/x['source'])==x['source_sha256']for x in build['compile'])
  env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSV41_VERIFY_MOE_WIDE=flag,LD_PRELOAD=str(P/'fault.so'),DSV41_TRACE_FILE=str(P/'baseline-prefix-faults.json'),LD_LIBRARY_PATH=str(P/'candidate')+':'+str(R/'build-aliceai/bin'))
  d=P/('baseline-prefix-'+flag);d.mkdir();command=[str(P/'baseline-prefix-harness'),str(d),str(P/'precision-inputs.json'),*cmd];arm={'flag':flag,'command':command,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'complete':False};j['arms'].append(arm);save(dest,j);halt=threading.Event();floor=[];mon=None;proc=None;watcher=None
  try:
   with(d/'server.log').open('wb')as log:
    proc=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);arm['pid']=proc.pid;save(dest,j);mon=MemoryMonitor(proc.pid)
    def watch():
     while not halt.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for phase in ['load','inference']for x in mon.stats[phase].values()if isinstance(x,dict))
      if bad:
       floor.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    watcher=threading.Thread(target=watch,daemon=True);watcher.start();start=time.monotonic();ready=False
    while proc.poll()is None:
     txt=(d/'server.log').read_text(errors='replace')
     if not ready and '\nREADY\n'in txt:
      maps=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert maps and all(str(P/'candidate')in x for x in maps);arm['loaded_llama_maps']=maps;ready=True;mon.inference();save(dest,j);print('READY_PROBE',flag,flush=True)
     assert time.monotonic()-start<3600;time.sleep(1)
   arm['exit_code']=proc.returncode;assert proc.returncode==0,proc.returncode;res=json.loads((d/'results.json').read_text());assert res['complete']and len(res['cases'])==3;arm['complete']=True;arm['result_sha256']=sha(d/'results.json');ref=json.loads((R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts/precision-0/results.json').read_text())['cases'][0]['result']['first8K'];arm['canonical_reference_exact']=[x['result']==ref for x in res['cases']];print('BASELINE_PREFIX_DONE',flag,arm['canonical_reference_exact'],flush=True)
  except Exception as ex:arm['error']=repr(ex);arm['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=45)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();arm['forced_kill']=True
   halt.set()
   if watcher:watcher.join(timeout=3)
   if mon:arm['memory']=mon.finish()
   arm['floor_breach']=floor;arm['exit_code']=proc.returncode if proc else None;save(dest,j)
  assert not floor and not arm.get('forced_kill')
 j['complete']=True;save(dest,j);print('ROLLBACK_PROBES_COMPLETE',flush=True)
if __name__=='__main__':main()
