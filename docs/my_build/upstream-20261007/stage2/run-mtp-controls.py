#!/usr/bin/env python3
"""Compare three-slot MTP behavior against isolated stage 1 libraries."""
import importlib.util
import json
import time
from pathlib import Path
root=Path('/home/sergei/_my_sync/llama_upstream_review/stage2')
while True:
    try:
        if len(json.loads((root/'corrected-control-results.json').read_text())) == 2:
            break
    except (OSError,json.JSONDecodeError):
        pass
    time.sleep(5)
spec=importlib.util.spec_from_file_location('parallel_check',root/'run-parallel.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results=[]
for profile in [('glm-mtp-separated-before','glm5next','draft-mtp',None,False),
                ('qwen-mtp-separated-before','qwen4exp','draft-mtp',None,False)]:
    results.append(module.run(*profile,before=True))
    (root/'mtp-before-summary.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)
