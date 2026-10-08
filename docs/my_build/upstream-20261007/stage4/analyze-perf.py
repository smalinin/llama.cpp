#!/usr/bin/env python3
import json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
results=[]
for gpu in ('4','3'):
    rows={}
    for version in ('before','after'):
        for repeat in range(3):
            text=(R/f'mat-perf-{version}-gpu{gpu}-r{repeat}.log').read_text()
            text=re.sub(r'\x1b\[[0-9;]*m','',text)
            cases=re.findall(r'MUL_MAT\(name=([^,]+),.*?([0-9.]+) us/run',text,re.S)
            if len(cases)!=8:raise RuntimeError(f'incomplete {gpu} {version} {repeat}: {len(cases)}')
            for name,time in cases:rows.setdefault(name,{}).setdefault(version,[]).append(float(time))
    for name,values in rows.items():
        old=statistics.median(values['before']);new=statistics.median(values['after'])
        results.append({'gpu_index':int(gpu),'case':name,'before_us':old,'after_us':new,'speedup':old/new,'time_change_percent':100*(new/old-1),'samples':values})
(R/'mat-performance.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:print(r['gpu_index'],r['case'],r['before_us'],r['after_us'],round(r['speedup'],2),round(r['time_change_percent'],1))
