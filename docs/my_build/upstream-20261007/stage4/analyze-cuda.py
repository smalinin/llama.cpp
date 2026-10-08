#!/usr/bin/env python3
from pathlib import Path
import json,array,csv,io,statistics
R=Path(__file__).resolve().parent
records={}
for label in ['check-'+v+'-gpu'+g for v in ('before','after') for g in ('4','3')]+['check-after-unfused-gpu4']:
    records[label]=[json.loads(l) for l in (R/(label+'.log')).read_text().splitlines() if l.startswith('{')]
    if len(records[label])!=528 or not all(x['passed'] for x in records[label]):raise RuntimeError(label+' incomplete or failed')
comparisons=[]
for gpu in ('4','3'):
    comparisons_for_gpu=[]
    a=records['check-before-gpu'+gpu];b=records['check-after-gpu'+gpu]
    da=array.array('f');da.frombytes((R/('check-before-gpu'+gpu+'.bin')).read_bytes())
    db=array.array('f');db.frombytes((R/('check-after-gpu'+gpu+'.bin')).read_bytes())
    for x,y in zip(a,b):
        assert x['case']==y['case'] and x['count']==y['count'] and x['offset']==y['offset']
        i=x['offset']//4;n=x['count'];va=da[i:i+n];vb=db[i:i+n]
        ia=array.array('I');ia.frombytes(va.tobytes());ib=array.array('I');ib.frombytes(vb.tobytes());diff=sum(u!=v for u,v in zip(ia,ib))
        comparisons_for_gpu.append({'case':x['case'],'bitwise_equal':diff==0,'different_values':diff,'max_abs':max((abs(u-v) for u,v in zip(va,vb)),default=0.0),'before_cpu_nmse':x['nmse'],'after_cpu_nmse':y['nmse']})
    comparisons.append({'gpu_index':int(gpu),'cases':len(a),'passed_cpu_reference_before':sum(x['passed'] for x in a),'passed_cpu_reference_after':sum(x['passed'] for x in b),'norm_bitwise_equal':sum(x['bitwise_equal'] for x in comparisons_for_gpu if x['case'].startswith('norm')),'mat_bitwise_equal':sum(x['bitwise_equal'] for x in comparisons_for_gpu if not x['case'].startswith('norm')),'details':comparisons_for_gpu})
a=(R/'check-after-gpu4.bin').read_bytes();b=(R/'check-after-unfused-gpu4.bin').read_bytes()
fusion=[]
for x in records['check-after-gpu4'][:144]:
    i=x['offset'];n=4*x['count'];fusion.append({'case':x['case'],'bitwise_equal':a[i:i+n]==b[i:i+n]})
def csv_rows(path,header):
    lines=path.read_text().splitlines();idx=next(i for i,l in enumerate(lines) if l.startswith(header));return list(csv.DictReader(io.StringIO('\n'.join(lines[idx:]))))
traces={}
for mode in ('norm','mmvf'):
    traces[mode]={}
    for version in ('before','after'):
        rows=csv_rows(R/(mode+'-'+version+'-stats.log'),'Time (%)')
        rows=[r for r in rows if r.get('Instances')]
        traces[mode][version]={'kernel_launches':sum(int(r['Instances']) for r in rows),'total_kernel_ns':sum(int(r['Total Time (ns)']) for r in rows),'graphs_evaluated':503,'kernels':rows}
result={'cpu_reference_checks':sum(len(v) for v in records.values()),'before_after':comparisons,'fusion_disable_control':fusion,'traces':traces}
(R/'cuda-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cpu_checks':result['cpu_reference_checks'],'comparisons':[{k:v for k,v in c.items() if k!='details'} for c in comparisons],'fusion_control_equal':sum(x['bitwise_equal'] for x in fusion),'traces':{m:{v:{k:x for k,x in d.items() if k!='kernels'} for v,d in ds.items()} for m,ds in traces.items()}},indent=2))
