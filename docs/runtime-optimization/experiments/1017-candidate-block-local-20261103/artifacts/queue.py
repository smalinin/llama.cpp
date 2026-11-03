from pathlib import Path
import time,json,subprocess,shutil
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');start=time.monotonic()
while True:
 j=json.load(open(P/'precision-results.json'))
 assert not any(a.get('error')for a in j['arms']),j
 if j['complete']:break
 assert time.monotonic()-start<7200,'precision timeout'
 time.sleep(2)
assert json.load(open(P/'operator-results.json'))['complete']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
# Initial metadata overlaps the external operator; preserve it, then refresh idle free-memory inputs for admission.
d=P/'metadata-initial';d.mkdir()
for pat in ['metadata-*.json','meta-trace-*.jsonl','metadata-*.log']:
 for f in list(P.glob(pat)):shutil.move(str(f),d/f.name)
for script in ['run-metadata.py','check-metadata.py','bench.py','admit.py','run-payload.py']:
 print('START',script,flush=True)
 with(P/(script+'.driver.log')).open('wb')as f:subprocess.run(['python3',str(P/script)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,flush=True)
print('ALL_GPU_GATES_DONE',flush=True)
