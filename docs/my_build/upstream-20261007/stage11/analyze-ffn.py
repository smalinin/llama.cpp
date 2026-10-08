#!/usr/bin/env python3
from array import array
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct

ROOT = Path(__file__).resolve().parent

def read(p, n=None):
    a = array('f')
    a.frombytes(p.read_bytes())
    return a if n is None else a[:n]

def compare(a, b):
    assert len(a) == len(b) and len(a)
    d = [float(x)-float(y) for x, y in zip(a,b)]
    assert all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
    return {'max_abs': max(map(abs,d)), 'rms': math.sqrt(sum(x*x for x in d)/len(d)),
            'bit_identical': a.tobytes() == b.tobytes()}

def difference(a,b):
    return array('f',(x-y for x,y in zip(a,b)))

def q8_fields(path):
    data=path.read_bytes()
    q, d, s = [],[],[]
    for slot in range(6):
        for block in range(72):
            off=(slot*80+block)*36
            scale, total=struct.unpack_from('<ee',data,off)
            d.append(scale); s.append(total)
            q.extend(struct.unpack_from('<32b',data,off+4))
    return q,d,s

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

summary={'capture_reproduction':{},'identical_input':{},'no_fusion':{},'frozen_down':{},'q8_capture':{},'cpu_delta':{}}
for arch in ['ada','ampere']:
    p=ROOT/('ffn-'+arch)
    for w in [1,2,4]:
        label=f'inputw{w}-batch{w}-'+('repeat' if w==1 else 'actual')
        summary['capture_reproduction'][f'{arch}-w{w}']=compare(read(p/(label+'-out.f32')),read(ROOT/f'inputs/captured-w{w}.f32'))
    for source in [1,2,4]:
        for w in [2,4]:
            values={}
            for kind,n in [('out',5120),('moe',5120),('moe-hidden',2304*6),('shared-hidden',2304),('router',384),('weights',6)]:
                values[kind]=compare(read(p/f'inputw{source}-batch1-repeat-{kind}.f32',n),read(p/f'inputw{source}-batch{w}-repeat-{kind}.f32',n))
            summary['identical_input'][f'{arch}-input{source}-w{w}']=values
    down=ROOT/('down-'+arch)
    for hidden in [1,2,4]:
        for w in [2,4]:
            summary['frozen_down'][f'{arch}-hidden{hidden}-w{w}']=compare(read(down/f'hiddenw{hidden}-batch1.f32',5120),read(down/f'hiddenw{hidden}-batch{w}.f32',5120))
    q1,d1,s1=q8_fields(ROOT/f'q8-{arch}/hiddenw1.q8_1')
    for w in [2,4]:
        qw,dw,sw=q8_fields(ROOT/f'q8-{arch}/hiddenw{w}.q8_1')
        changed=[i for i,(a,b) in enumerate(zip(q1,qw)) if a!=b]
        cpu_q,cpu_d=q8_cpu(read(p/f'inputw1-batch{w}-repeat-moe-hidden.f32',2304*6))
        assert cpu_q==qw and cpu_d==dw
        summary['q8_capture'][f'{arch}-w{w}']={'changed_integer_indices':changed,
            'integer_changes':[{'index':i,'slot':i//2304,'offset':i%2304,'from':q1[i],'to':qw[i]} for i in changed],
            'changed_half_d_blocks':[i for i,(a,b) in enumerate(zip(d1,dw)) if a!=b],
            'changed_half_sum_blocks':[i for i,(a,b) in enumerate(zip(s1,sw)) if a!=b],
            'cpu_equations_match_q_and_d': True}
        gpu_delta=difference(read(down/f'hiddenw{w}-batch1.f32',5120),read(down/'hiddenw1-batch1.f32',5120))
        cpu_delta=read(ROOT/f'cpu-delta-{arch}/w{w}-q8-delta.f32')
        float_delta=read(ROOT/f'cpu-delta-{arch}/w{w}-float-delta.f32')
        full_delta=difference(read(p/f'inputw1-batch{w}-repeat-moe.f32',5120),read(p/'inputw1-batch1-repeat-moe.f32',5120))
        zero=array('f',[0]*5120)
        summary['cpu_delta'][f'{arch}-w{w}']={'frozen_gpu_delta':compare(gpu_delta,zero),
            'predicted_q8_delta':compare(cpu_delta,zero), 'q8_prediction_residual':compare(gpu_delta,cpu_delta),
            'q8_prediction_residual_full_moe':compare(full_delta,cpu_delta), 'float_activation_delta':compare(float_delta,zero)}
    rows=[json.loads(l) for l in (p/'results.jsonl').read_text().splitlines()]
    assert len(rows)==11 and all(r['repeat_stable'] and r['column_max']==0 for r in rows)
    rows=[json.loads(l) for l in (down/'results.jsonl').read_text().splitlines()]
    assert len(rows)==9 and all(r['repeat_stable'] and r['column_max']==0 for r in rows)

for w in [1,2,4]:
    assert (ROOT/f'q8-ada/hiddenw{w}.q8_1').read_bytes()==(ROOT/f'q8-ampere/hiddenw{w}.q8_1').read_bytes()
for w in [2,4]:
    p=ROOT/'ffn-ada-no-fusion'
    summary['no_fusion'][f'w{w}']=compare(read(p/'inputw1-batch1-repeat-out.f32',5120),read(p/f'inputw1-batch{w}-repeat-out.f32',5120))
assert all(x['bit_identical'] for x in summary['capture_reproduction'].values())
assert all(x['changed_integer_indices']==[5121,6232] and not x['changed_half_d_blocks'] for x in summary['q8_capture'].values())
assert all(x['q8_prediction_residual']['max_abs']<5e-8 for x in summary['cpu_delta'].values())
# The profile must preserve all outputs of the unprofiled process.
for p in (ROOT/'ffn-ada').glob('*'):
    assert p.read_bytes()==(ROOT/'ffn-ada-profile'/p.name).read_bytes()
summary['profile_outputs_identical']=True
(ROOT/'ffn-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

c=sqlite3.connect(ROOT/'ffn-ada-profile.sqlite')
rows=c.execute('''select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k
                  join StringIds s on s.id=k.demangledName group by s.value order by count(*) desc''').fetchall()
launches=c.execute('''select s.value,k.blockX,k.blockY,k.blockZ,k.gridX,k.gridY,k.gridZ,count(*)
                     from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
                     where s.value like '%mul_mat_vec_q%' group by 1,2,3,4,5,6,7''').fetchall()
profile={'note':'Kernel identification only; aggregate GPU duration is not model throughput',
         'kernels':[dict(name=r[0],calls=r[1],gpu_ms=r[2]) for r in rows],
         'launches':[dict(name=r[0],block=list(r[1:4]),grid=list(r[4:7]),calls=r[7]) for r in launches]}
(ROOT/'kernel-summary.json').write_text(json.dumps(profile,indent=2)+'\n')
print(json.dumps({'capture_exact':len(summary['capture_reproduction']), 'ada_w2':summary['identical_input']['ada-input1-w2'],
                  'frozen_down':summary['frozen_down']['ada-hidden1-w2'], 'q8':summary['q8_capture']['ada-w2'],
                  'cpu_delta':summary['cpu_delta']['ada-w2']},indent=2))

import statistics
bench={}
for arch in ['ada','ampere']:
    rows=[json.loads(l) for l in (ROOT/f'bench-{arch}/timings.jsonl').read_text().splitlines()]
    assert len(rows)==15 and all(r['iterations']==100 and r['mean_ms']>0 for r in rows)
    median={w:statistics.median(r['mean_ms'] for r in rows if r['width']==w) for w in [1,2,4]}
    bench[arch]={'median_ms_per_call':median,
                 'estimated_full_scalarization_cost_ratio':{w:w*median[1]/median[w] for w in [2,4]}}
bench['note']='Synchronized host wall time of isolated full FFN, 10 warmups and 5 groups of 100 calls per width; scalarization estimates use width*T1/Twidth. No full-model TPS measurement; replacing only routed up/gate has not been benchmarked.'
(ROOT/'benchmark-summary.json').write_text(json.dumps(bench,indent=2)+'\n')
print(json.dumps(bench,indent=2))
