#!/usr/bin/env python3
"""Repeat the native GLM-DSA performance comparison with six greedy trials."""
from pathlib import Path
import argparse,time,json
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--wait-for',type=int);args=parser.parse_args()
if args.wait_for:
    command=Path(f'/proc/{args.wait_for}/cmdline')
    while True:
        try:
            active=b'run-real.py' in command.read_bytes()
        except OSError:
            break
        if not active:break
        time.sleep(2)
    print('Main model checks finished; starting GLM-DSA repeat control',flush=True)
driver=(R/'run-real.py').read_text();prefix=driver[:driver.index("parser=argparse.ArgumentParser();parser.add_argument('--models'")]
setup={'__name__':'validation_setup','__file__':str(R/'run-real.py')};exec(compile(prefix,str(R/'run-real.py'),'exec'),setup)
source=setup['source'].replace("[('greedy-1', 0), ('greedy-2', 0), ('sampling-1', 0.8), ('sampling-2', 0.8)]","[('warmup',0)]+[(f'greedy-{i}',0) for i in range(1,7)]")
source=source.replace("for mode in ('greedy', 'sampling'):","for mode in ('greedy',):")
namespace={'__name__':'dsa_validation'};exec(compile(source,'dsa_validation','exec'),namespace)
results=[]
for version in ('before','after'):
    out=R/('validation-'+version);out.mkdir(exist_ok=False)
    result=namespace['run_case']('glm-dsa',False,out,R/(version+'-bin/llama-server'));result['binary_set']=version;results.append(result)
    namespace['save'](R/'dsa-validation-progress.json',results)
    if result['status']!='passed':raise SystemExit(1)
