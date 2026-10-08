#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
limit=time.monotonic()+1800
while True:
    p=ROOT/'controls-manifest.json'
    if p.exists() and 'exit_code' in json.loads(p.read_text()): break
    if time.monotonic()>limit: raise TimeoutError('controls not finished')
    time.sleep(1)
commands=[
    [sys.executable,str(ROOT/'run-target-controls.py'),'--no-fusion','--modes','default'],
    [sys.executable,str(ROOT/'run-current-server.py'),'--configs','off','n1-p0',
     '--diagnostic-repeats','2','--output',str(ROOT/'n1-server')],
]
for command in commands:
    print('RUN',command,flush=True)
    result=subprocess.run(command)
    if result.returncode: raise SystemExit(result.returncode)
