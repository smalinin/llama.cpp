from pathlib import Path
import json,subprocess,time,os
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp')
started=time.monotonic()
while True:
 try:
  m=json.load(open(P/'long-manifest.json'))
  if m.get('exit_code')is not None:
   assert m.get('complete')and m['exit_code']==0,m.get('error')
   break
 except json.JSONDecodeError:pass
 assert time.monotonic()-started<3700,'long gate timeout'
 time.sleep(2)
subprocess.run(['python3',str(P/'check-long.py')],check=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
e=os.environ.copy();e.update(CUDA_DEVICE_ORDER='FASTEST_FIRST',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',LD_LIBRARY_PATH=str(R/'build-aliceai/bin'),GGML_CUDA_GRAPH_STATS='1')
with(P/'scheduler-counted.log').open('wb')as log:
 subprocess.run([str(P/'scheduler-counted'),str(P/'scheduler-counted-results.json')],cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,check=True)
r=json.load(open(P/'scheduler-counted-results.json'));assert r['complete']and r['capture_outputs_per_mode']==32
print('COUNTED_SCHEDULER_PASS',flush=True)
subprocess.run(['python3',str(P/'bench.py')],cwd=R,check=True)
