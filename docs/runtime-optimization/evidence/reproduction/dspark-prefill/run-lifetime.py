from pathlib import Path
import json,os,hashlib,subprocess,time
from memory_monitor import MemoryMonitor
P=Path(__file__).resolve().parent;R=Path('/home/sergei/Github/llama.cpp');q=json.load(open('/tmp/dsv41-feature-local-joint-20261103/prepared.json'));a=q['base_args'];args=[];i=0
while i<len(a):
 k=a[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 args.append(k)
 if k not in ['--no-warmup','--no-mmproj-auto','--no-spec-draft-backend-sampling']:args.append(a[i+1]);i+=2
 else:i+=1
args[args.index('--ctx-size')+1]='100352'
e={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV41_'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};e.update(q['environment']);e.pop('DSV41_SHARED64_PERF',None)
cmd=[str(P/'lifetime'),str(P/'lifetime-native'),str(P/'inputs.json'),*args]
j={'complete':False,'command':cmd,'environment':{k:v for k,v in e.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV41_','LD_'))},'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'patch':subprocess.check_output(['git','diff'],cwd=R,text=True),'hashes':{n:hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()for n in ['llama-server','libllama.so','libllama-common.so','libggml-cuda.so']}}
def save():(P/'lifetime-manifest.json').write_text(json.dumps(j,indent=2)+'\n')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
proc=None;mon=None;save();started=time.monotonic();ready=False
try:
 with(P/'lifetime.log').open('wb')as f:
  proc=subprocess.Popen(cmd,env=e,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);j['pid']=proc.pid;mon=MemoryMonitor(proc.pid);save()
  while proc.poll()is None:
   log=(P/'lifetime.log').read_text(errors='replace')
   if not ready and '\nREADY\n'in log:ready=True;mon.inference();print('LIFETIME_READY',flush=True)
   if ready and min(x['free']for x in mon.sample())<512*1048576:raise RuntimeError('VRAM floor512 breached')
   assert time.monotonic()-started<1800,'timeout';time.sleep(1)
  j['exit_code']=proc.returncode;assert proc.returncode==0,'native exit '+str(proc.returncode)
  assert json.load(open(P/'lifetime-native/results.json'))['complete'];j['complete']=True
finally:
 if proc and proc.poll()is None:proc.terminate();proc.wait(timeout=60)
 if mon:j['memory']=mon.finish()
 if proc:j['exit_code']=proc.returncode
 j['seconds']=time.monotonic()-started;save()
print('LIFETIME_DONE',flush=True)
