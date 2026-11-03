from pathlib import Path
import os,json,subprocess,time,hashlib,signal,re
from memory_monitor import MemoryMonitor
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp')
cfg=json.load(open(P/'config.json'));assert json.load(open(P/'metadata-results.json'))['complete']
args=cfg['command'][1:];kept=[];i=0
while i<len(args):
 k=args[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 kept.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:kept.append(args[i+1]);i+=2
kept+=['--no-spec-draft-backend-sampling']
kept[kept.index('--spec-draft-n-max')+1]='5'
kept[kept.index('--spec-draft-p-min')+1]='0'
env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']}
env.update(cfg['environment']);env.update(LD_PRELOAD=str(P/'trace.so'),LLAMA_DSV41_CANDIDATE_POOL_LOCAL='1',DSV41_TRACE_FILE=str(P/'fresh-on-trace.json'))
cmd=[str(P/'fresh-on'),str(P/'fresh-on-native'),str(P/'inputs.json'),*kept]
j={'complete':False,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'head':cfg['source_head'],'runtime_patch':cfg['runtime_patch'],'library_hashes':cfg['source_library_hashes'],'helper_sha256':{n:hashlib.file_digest((P/n).open('rb'),'sha256').hexdigest()for n in ['fresh-on','trace.so']},'scope':'Fresh constantON constructor K5 exact lifecycle controls, compare to prior OFF results externally, HC sharing defaultON under fixed default M2/LOCAL. Two corpora98K,raw target/draft logits/state/taps,K3K5,append1/7/511/513,restore/rollback,128 NLL labels/corpus/mode,controlled seed failure+retry. No speed claim.'}
def save():(P/'fresh-on-manifest.json').write_text(json.dumps(j,indent=2)+'\n')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
for n,h in j['library_hashes'].items():assert hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()==h
proc=None;mon=None;started=time.monotonic();save()
try:
 with(P/'fresh-on.log').open('wb')as log:
  proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);j['pid']=proc.pid;mon=MemoryMonitor(proc.pid);ready=False;save()
  while proc.poll()is None:
   text=(P/'fresh-on.log').read_text(errors='replace')
   if not ready and '\nREADY\n'in text:ready=True;mon.inference();print('LONG_READY',flush=True)
   with mon.lock:
    breach=any(v.get('min_free',512*1048576)<512*1048576 for phase in ['load','inference']for v in mon.stats[phase].values()if isinstance(v,dict))
   if breach:j['floor_breach']=True;raise RuntimeError('sampled floor512 breached')
   assert time.monotonic()-started<3600,'timeout'
   time.sleep(.2)
  j['exit_code']=proc.returncode;assert proc.returncode==0,'native exit '+str(proc.returncode)
  result=json.load(open(P/'long-native/results.json'));assert result['complete']and len(result['cases'])==12 and len(result['faults'])==2
  j['complete']=True
except Exception as e:j['error']=repr(e);raise
finally:
 if proc and proc.poll()is None:
  proc.send_signal(signal.SIGTERM)
  try:proc.wait(timeout=60)
  except subprocess.TimeoutExpired:proc.kill();proc.wait();j['forced_kill']=True
 if mon:j['memory']=mon.finish()
 j['exit_code']=None if proc is None else proc.returncode;j['seconds']=time.monotonic()-started;save()
print('LONG_DONE',j['seconds'],flush=True)
