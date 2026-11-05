from pathlib import Path
import json,time,subprocess
P=Path(__file__).resolve().parent
assert P.parent.name=='1017-verify-float6-acceptance-20261106'
t0=time.monotonic()
while True:
 f=P/'lifecycle-results.json'
 if f.exists():
  d=json.loads(f.read_text())
  assert not any('error'in x for x in d.get('arms',[])), 'Lifecycle failed; native stage not started'
  if d.get('complete'):
   assert d['all_groups_exact']and d['dispatch_reduction_verified']
   break
 assert time.monotonic()-t0<10800
 time.sleep(2)
r=subprocess.run(['python3','-u',str(P/'native-run.py')],cwd=P)
raise SystemExit(r.returncode)
