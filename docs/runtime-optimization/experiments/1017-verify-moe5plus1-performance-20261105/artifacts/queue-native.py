from pathlib import Path
import json,time,subprocess,sys
P=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-verify-moe5plus1-performance-20261105/artifacts');start=time.monotonic()
while True:
 p=P/'width-results.json'
 if p.exists():
  j=json.loads(p.read_text())
  if any(a.get('error')for a in j.get('arms',[])):raise RuntimeError('width stage failed; native not launched')
  if j.get('complete'):
   assert j['all_outputs_state_exact'];break
 assert time.monotonic()-start<3600;time.sleep(2)
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
 assert time.monotonic()-start<3660;time.sleep(1)
print('WIDTH_GATE_COMPLETE_START_NATIVE',flush=True)
subprocess.run([sys.executable,str(P/'native-run.py')],cwd='/home/sergei/Github/llama.cpp',check=True)
