from pathlib import Path
import os,subprocess,json,hashlib,time,sys,shutil,threading,signal
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();cfg=json.loads((P/'config.json').read_text());env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env['LD_LIBRARY_PATH']=str(R/'build-aliceai/bin');cmd=[str(P/'operator-harness'),str(P/'operator-results.json')];start=time.monotonic();mon=None;proc=None;breach=[]
try:
 with(P/'operator.log').open('wb')as log:
  proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);mon=MemoryMonitor(proc.pid);mon.inference()
  while proc.poll()is None:
   with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for ph in ['load','inference']for x in mon.stats[ph].values()if isinstance(x,dict))
   if bad:breach.append(time.time());proc.send_signal(signal.SIGTERM);break
   assert time.monotonic()-start<900;time.sleep(.25)
  proc.wait(timeout=60)
finally:
 if proc and proc.poll()is None:proc.send_signal(signal.SIGTERM);proc.wait(timeout=60)
 memory=mon.finish()if mon else{}
 j={'complete':False,'command':cmd,'elapsed_s':time.monotonic()-start,'exit_code':proc.returncode if proc else None,'memory':memory,'floor_breach':breach};(P/'operator-run.json').write_text(json.dumps(j,indent=2)+'\n')
r=json.loads((P/'operator-results.json').read_text());assert proc.returncode==0 and r['complete']and len(r['cases'])==360 and not breach;assert all(x['exact']and x['max_abs']==0 for x in r['cases']);j['complete']=True;j['exact_evaluations']=360;j['distinct_configurations']=180;(P/'operator-run.json').write_text(json.dumps(j,indent=2)+'\n');print('FLOAT6_360_EXACT',round(j['elapsed_s'],2))
