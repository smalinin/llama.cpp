#!/usr/bin/env python3
from array import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT=Path(__file__).resolve().parent

def read(p,n=None):
    a=array('f');a.frombytes(p.read_bytes())
    return a if n is None else a[:n]

def compare(a,b):
    assert len(a)==len(b) and a
    delta=[x-y for x,y in zip(a,b)]
    assert all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
    return {'max_abs':max(map(abs,delta)),'rms':math.sqrt(sum(x*x for x in delta)/len(delta)),
            'bit_identical':a.tobytes()==b.tobytes()}

def q8_cpu(a):
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    q,d=[],[]
    for start in range(0,len(a),32):
        group=a[start:start+32]
        scale=f32(max(map(abs,group))/127)
        d.append(struct.unpack('<e',struct.pack('<e',scale))[0])
        for x in group:
            v=f32(x/scale) if scale else 0
            q.append(math.floor(v+.5) if v>=0 else math.ceil(v-.5))
    return q,d

summary={'native_matches_stage11':{},'scalar_identity':{},'identical_input':{},'actual_inputs':{},'benchmark':{},
         'note':'Scalarization changes only routed up/gate/SwiGLU. Routing weights, shared expert, down and final reduction keep native batch. Benchmarks are isolated FFN host wall time, not model TPS.'}
for arch in ['ada','ampere']:
    p=ROOT/('controls-'+arch)
    rows=[json.loads(l) for l in (p/'results.jsonl').read_text().splitlines()]
    assert len(rows)==22 and all(x['repeat_stable'] and x['column_max']==0 for x in rows)
    stage11=ROOT.parent/('stage11/ffn-'+arch)
    nfiles=0
    for f in p.glob('native-*.f32'):
        assert f.read_bytes()==(stage11/f.name[len('native-'):]).read_bytes()
        nfiles+=1
    for f in p.glob('native-*.i32'):
        assert f.read_bytes()==(stage11/f.name[len('native-'):]).read_bytes()
        nfiles+=1
    summary['native_matches_stage11'][arch]=nfiles
    for source in [1,2,4]:
        for kind,n in [('out',5120),('moe',5120),('moe-hidden',2304*6),('shared-hidden',2304),('router',384),('weights',6)]:
            a=read(p/f'native-inputw{source}-batch1-repeat-{kind}.f32',n)
            b=read(p/f'scalar-upgate-inputw{source}-batch1-repeat-{kind}.f32',n)
            result=compare(a,b)
            assert result['bit_identical']
            summary['scalar_identity'][f'{arch}-source{source}-{kind}']=result
        for width in [2,4]:
            values={}
            for kind,n in [('out',5120),('moe',5120),('moe-hidden',2304*6),('shared-hidden',2304),('router',384),('weights',6)]:
                a=read(p/f'scalar-upgate-inputw{source}-batch1-repeat-{kind}.f32',n)
                b=read(p/f'scalar-upgate-inputw{source}-batch{width}-repeat-{kind}.f32',n)
                values[kind]=compare(a,b)
                if kind=='moe-hidden':
                    assert values[kind]['bit_identical'] and q8_cpu(a)==q8_cpu(b)
            assert values['out']['max_abs']<=6e-8
            summary['identical_input'][f'{arch}-source{source}-w{width}']=values
    for width in [2,4]:
        a=read(p/'scalar-upgate-inputw1-batch1-repeat-out.f32',5120)
        b=read(p/f'scalar-upgate-inputw{width}-batch{width}-actual-out.f32',5120)
        summary['actual_inputs'][f'{arch}-w{width}']=compare(a,b)
    for f in p.glob('scalar-upgate-*-ids.i32'):
        assert f.read_bytes()==(p/f.name.replace('scalar-upgate-','native-',1)).read_bytes()
    times=[json.loads(l) for l in (p/'timings.jsonl').read_text().splitlines()]
    assert len(times)==30 and all(x['iterations']==100 and x['mean_ms']>0 for x in times)
    med={m:{w:statistics.median(x['mean_ms'] for x in times if x['mode']==m and x['width']==w) for w in [1,2,4]} for m in ['native','scalar-upgate']}
    summary['benchmark'][arch]={'median_ms_per_call':med,
        'cost_ratio':{w:med['scalar-upgate'][w]/med['native'][w] for w in [1,2,4]}}
(ROOT/'controls-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'native_control_files':summary['native_matches_stage11'],
                  'ada_w2':summary['identical_input']['ada-source1-w2'],
                  'actual_inputs':summary['actual_inputs'],'benchmark':summary['benchmark']},indent=2))
