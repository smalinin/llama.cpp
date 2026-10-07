#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
S=Path('/home/sergei/_my_sync/llama_upstream_review/stage3')
while True:
    try:
        results=json.loads((S/'continuation-runs/summary.json').read_text())
        if len(results)==4:break
    except (OSError,json.JSONDecodeError):pass
    time.sleep(5)
os.environ['STAGE3_QWEN_Q8']='1';os.environ['STAGE3_CONTINUATION']='1'
outer={'__name__':'continuation_runner'}
exec(compile((S/'run-real-state.py').read_text().split('\nimport argparse')[0],'run-real-state.py','exec'),outer)
source=outer['source']
source=source.replace("[('greedy-1', 0), ('greedy-2', 0), ('sampling-1', 0.8), ('sampling-2', 0.8)]", "[('greedy-1', 0)]")
a=source.index('            import struct');b=source.index("            if os.getenv('STAGE3_CONTINUATION'):",a)
source=source[:a]+source[b:]
source=source.replace("allow_full_prefill=mtp)","allow_full_prefill=mtp,prior_response='greedy-1-response.json')")
source=source.replace("'n_predict': 128", "'n_predict': 32")
a=source.index("            for mode in ('greedy', 'sampling'):");b=source.index('        except Exception as error:',a)
source=source[:a]+source[b:]
namespace={'__name__':'continuation_check'}
exec(compile(source,'continuation-check','exec'),namespace)
out=S/'continuation-runs-q8';out.mkdir(exist_ok=False)
result=namespace['run_case']('qwen4exp',False,out,S/'bin/llama-server')
namespace['save'](out/'summary.json',[result])
raise SystemExit(0 if result['status']=='passed' else 1)
