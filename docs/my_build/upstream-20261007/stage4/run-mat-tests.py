#!/usr/bin/env python3
import json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;results=[]
for gpu in ('4','3'):
    for version in (() if '--perf-only' in sys.argv else ('before','after')):
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,LD_LIBRARY_PATH=str(R/(version+'-bin')))
        label=f'mat-{version}-gpu{gpu}'
        command=[str(R/'cuda-check'),'strides',str(R/(label+'.bin'))]
        with (R/(label+'.log')).open('w') as log:rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        results.append({'case':label,'exit_code':rc,'command':command});print(label,rc,flush=True)
        if rc:raise SystemExit(rc)
    for repeat in range(3):
        for version in (('before','after') if repeat%2==0 else ('after','before')):
            env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,LD_LIBRARY_PATH=str(R/(version+'-bin')))
            label=f'mat-perf-{version}-gpu{gpu}-r{repeat}'
            command=[str(R/(version+'-bin/test-backend-ops')),'perf','-b','CUDA0','--test-file',str(R/'mat-perf-cases.txt'),'--output','console']
            with (R/(label+'.log')).open('w') as log:rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            results.append({'case':label,'exit_code':rc,'command':command});print(label,rc,flush=True)
            (R/'mat-test-results.json').write_text(json.dumps(results,indent=2)+'\n')
            if rc:raise SystemExit(rc)
