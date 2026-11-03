from pathlib import Path
import json,time,subprocess,os
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');start=time.monotonic()
while True:
 try:
  x=json.load(open(P/'fresh-on-manifest.json'))
  if x.get('exit_code')is not None:
   assert x['complete']and x['exit_code']==0,x.get('error');break
 except (FileNotFoundError,json.JSONDecodeError):pass
 assert time.monotonic()-start<3700;time.sleep(2)
subprocess.run(['python3',str(P/'check-fresh-on.py')],cwd=R,check=True)
subprocess.run(['python3',str(P/'check-long.py')],cwd=R,check=True)
subprocess.run(['python3',str(P/'finalize-guard.py')],cwd=R,check=True)
e={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};e.update(CUDA_DEVICE_ORDER='FASTEST_FIRST',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',LD_LIBRARY_PATH=str(R/'build-aliceai/bin'),GGML_CUDA_ARGSORT_RADIX='1');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with(P/'operator.log').open('wb')as f:subprocess.run([str(P/'operator'),str(P/'operator-results.json')],cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
print('OPERATOR_DONE',flush=True)
subprocess.run(['python3',str(P/'bench.py')],cwd=R,check=True)
subprocess.run(['python3',str(P/'admit.py')],cwd=R,check=True)
subprocess.run(['python3',str(P/'run-payload.py')],cwd=R,check=True)
print('ALL_GATES_DONE',flush=True)
