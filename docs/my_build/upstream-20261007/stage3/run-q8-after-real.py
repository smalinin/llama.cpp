#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path
root=Path('/home/sergei/_my_sync/llama_upstream_review/stage3')
while True:
    try:
        results=json.loads((root/'runs/summary.json').read_text())
        if len(results)==4: break
    except (OSError,json.JSONDecodeError): pass
    time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,str(root/'run-real-state.py'),'--qwen-q8']))
