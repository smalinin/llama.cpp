#!/usr/bin/env python3
from array import array
import importlib.util
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',ROOT/'analyze-chain.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
summary={'same_input':{},'actual_capture_controls':{},'cpu_double_error':{},'source_first_token_differences':{},'stable_cases':0,'isolated_processes':8}
for dataset in ['hc0','router0','post0','post2']:
 n={'hc0':20480,'router0':5120,'post0':5120,'post2':5120}[dataset];m={'hc0':24,'router0':384,'post0':20480,'post2':20480}[dataset]
 for arch in ['ada','ampere']:
  d=ROOT/f'{dataset}-{arch}';rs=[json.loads(l) for l in (d/'results.jsonl').read_text().splitlines()];assert len(rs)==11
  assert all(r['repeat_stable'] and r['column_max']==0 for r in rs);summary['stable_cases']+=len(rs)
  for source in [1,2,4]:
   base=read(d/f'source{source}-w1-repeat.f32');assert len(base)==m
   for width in [1,2,4]:
    a=read(d/f'source{source}-w{width}-repeat.f32');assert len(a)==m*width
    assert all(a[k*m:(k+1)*m].tobytes()==a[:m].tobytes() for k in range(width))
    summary['same_input'][f'{dataset}-{arch}-source{source}-w{width}']=c.compare(base,a[:m])
    if dataset in ['hc0','router0']:summary['cpu_double_error'][f'{dataset}-{arch}-source{source}-w{width}']=c.compare(read(d/f'cpu-source{source}.f32'),a[:m])
   p=d/(f'source{source}-w{source}-'+('repeat' if source==1 else 'actual')+'.f32')
   ref=ROOT/'inputs'/dataset/(f'captured-w{source}.f32' if dataset in ['hc0','router0'] else f'w{source}-output.f32')
   value=c.compare(read(ref),read(p));value.update(sha256=sha(p),reference_sha256=sha(ref))
   summary['actual_capture_controls'][f'{dataset}-{arch}-w{source}']=value
  summary['source_first_token_differences'][f'{dataset}-{arch}']={str(w):c.compare(read(d/'source1-w1-repeat.f32'),read(d/f'source{w}-w1-repeat.f32')) for w in [2,4]}
assert summary['stable_cases']==88
assert all(v['bit_identical'] for k,v in summary['actual_capture_controls'].items() if '-ada-' in k)
assert all(v['bit_identical'] for k,v in summary['same_input'].items() if k.startswith('post'))
(ROOT/'isolated-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'stable_cases':summary['stable_cases'],'actual_captures_exact':sum(v['bit_identical'] for v in summary['actual_capture_controls'].values()),'actual_captures_total':len(summary['actual_capture_controls']),'same_input_source1':{k:v for k,v in summary['same_input'].items() if 'source1' in k},'cpu_source1':{k:v for k,v in summary['cpu_double_error'].items() if 'source1' in k}},indent=2))
