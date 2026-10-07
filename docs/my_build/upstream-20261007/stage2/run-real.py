#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
ROOT=Path('/home/sergei/_my_sync/llama_upstream_review')
spec=importlib.util.spec_from_file_location('baseline',ROOT/'run_baseline.py')
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
output=ROOT/'stage2/runs';output.mkdir(exist_ok=True)
server=ROOT/'stage2/bin/llama-server'
results=[]
for name,speculative in [('glm5next',False),('glm5next',True),('qwen4exp',False),('qwen4exp',True),('glm-dsa',True),('deepseek41',True)]:
    results.append(baseline.run_case(name,speculative,output,server))
    baseline.save(output/'summary.json',results)
raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)
