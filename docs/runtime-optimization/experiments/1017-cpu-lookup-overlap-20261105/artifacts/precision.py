from pathlib import Path
import os,sys,json,hashlib,subprocess,time,threading,signal,re,traceback
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts';B=R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):q=p.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def main():
 while not(P/'precision-release').exists():time.sleep(.2)
 cfg=json.loads((B/'config.json').read_text());assert not(P/'precision-results.json').exists();native=json.loads((P/'native-results.json').read_text());assert native['complete'] and native['all_outputs_counters_exact']
 cmd=[];a=cfg['command'][1:];i=0
 while i<len(a):
  k=a[i]
  if k in ['--host','--port','--alias']:i+=2;continue
  cmd.append(k)
  if k in ['--no-warmup','--no-mmproj-auto']:i+=1
  else:cmd.append(a[i+1]);i+=2
 cmd+=['--no-spec-draft-backend-sampling'];cmd[cmd.index('--spec-draft-p-min')+1]='0';cmd[cmd.index('--spec-draft-n-max')+1]='5'
 j={'complete':False,'scope':'Fresh new-host OFF vs ON only, not historical arithmetic/state. Two98K text/code corpora,12result groups per arm, full target/draft logits/taps/state/restores/K3K5/partial/full/reject rollback,128teacher-forced NLL per corpus and two seed-fault/retry cases per arm. Transport-only exactness required. Backend sampling disabled for raw logits; not speed timing. No eval callback; small external fault interposer only.','candidate_sha256':sha(P/'candidate/libggml-base.so.0'),'corpus_sha256':sha(P/'precision-inputs.json'),'arms':[]};persist=lambda:save(P/'precision-results.json',j);persist();ref=None
 for flag in ['0','1']:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert sha(P/'candidate/libggml-base.so.0')==j['candidate_sha256']
  env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(GGML_SCHED_CPU_LOOKUP_EARLY=flag,LD_LIBRARY_PATH=str(P/'candidate')+':'+str(R/'build-aliceai/bin'),LD_PRELOAD=str(P/'fault.so'),DSV41_TRACE_FILE=str(P/('precision-'+flag+'-faults.json')))
  d=P/('precision-'+flag);d.mkdir();command=[str(P/'precision-harness'),str(d),str(P/'precision-inputs.json'),*cmd];arm={'flag':flag,'command':command,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'complete':False};j['arms'].append(arm);persist();proc=None;mon=None;halt=threading.Event();floor=[];watcher=None
  try:
   with(d/'server.log').open('wb')as log:
    proc=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);arm['pid']=proc.pid;persist();mon=MemoryMonitor(proc.pid)
    def watch():
     while not halt.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for phase in ['load','inference']for x in mon.stats[phase].values()if isinstance(x,dict))
      if bad:
       floor.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    watcher=threading.Thread(target=watch,daemon=True);watcher.start();start=time.monotonic();ready=False
    while proc.poll()is None:
     text=(d/'server.log').read_text(errors='replace')
     if not ready and '\nREADY\n'in text:ready=True;mon.inference();print('READY_PRECISION',flag,flush=True)
     assert time.monotonic()-start<3600;time.sleep(1)
    arm['exit_code']=proc.returncode;assert proc.returncode==0,('precision exit',flag,proc.returncode)
   result=json.loads((d/'results.json').read_text());assert result['complete']and len(result['cases'])==12 and len(result['faults'])==2
   rows={(r['corpus'],r['key']):r['result']for r in result['cases']}
   if ref is None:ref=rows
   else:
    mismatch=[{'corpus':k[0],'key':k[1]}for k in ref if rows[k]!=ref[k]];arm['mismatches']=mismatch;assert not mismatch,mismatch
   faults=json.loads((P/('precision-'+flag+'-faults.json')).read_text());assert faults['injected_failures']==2
   arm['complete']=True;arm['groups_exact']=len(rows);arm['faults']=faults
  except Exception as ex:arm['error']=repr(ex);arm['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();arm['forced_kill']=True
   halt.set()
   if watcher:watcher.join(timeout=3)
   if mon:arm['memory']=mon.finish()
   arm['floor_breach']=floor;arm['exit_code']=proc.returncode if proc else None;lines=(d/'server.log').read_text(errors='replace').splitlines();arm['warnings']=[x for x in lines if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',x,re.I)];arm['activation']=[x for x in lines if 'sched_cpu_lookup:'in x];persist()
  assert not floor and not arm['warnings']and not arm.get('forced_kill');assert flag=='0'or any(re.search('early_transitions=[1-9]',x)for x in arm['activation']);print('PRECISION_ARM_DONE',flag,flush=True)
 j['all_groups_exact']=True;j['complete']=True;persist();print('PRECISION_COMPLETE',flush=True)
if __name__=='__main__':main()
