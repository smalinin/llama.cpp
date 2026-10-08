from array import array
import hashlib,json,math,runpy,struct
from pathlib import Path

R=Path(__file__).resolve().parent
helpers=runpy.run_path(str(R/'analyze-chain.py'));compare=helpers['compare']
def load(p):
    a=array('f');a.frombytes(p.read_bytes());return a
result={'compressor':{},'attention':{}}
for arch in ['ada','ampere']:
    result['compressor'][arch]={}
    for name in ['kv','gate']:
        d=R/f'compressor-{name}-{arch}';inputs=R/'compressor-inputs'/name
        rows=[json.loads(l) for l in (d/'results.jsonl').read_text().splitlines()]
        assert len(rows)==11 and all(r['repeat_stable'] and r['column_max']==0 for r in rows)
        records={}
        for source in [1,2,4]:
            col=2 if source==4 else 0
            label=f'source{source}-w{source}-'+('repeat' if source==1 else 'actual')
            a=load(d/f'{label}.f32');capture=load(inputs/f'captured-w{source}.f32')
            assert a.tobytes()==capture.tobytes(),(arch,name,source)
            scalar=load(d/f'source{source}-w1-repeat.f32')
            cpu=load(d/f'cpu-source{source}.f32')
            selected=a[col*512:(col+1)*512]
            records[str(source)]={'capture_bit_identical':True,'vs_scalar':compare(scalar,selected),
                                  'scalar_vs_cpu_double':compare(cpu,scalar),'batch_vs_cpu_double':compare(cpu,selected)}
        for source in [2,4]:
            a=load(inputs/f'input-w{source}.f32');col=2 if source==4 else 0
            assert a[col*5120:(col+1)*5120].tobytes()==load(inputs/'input-w1.f32').tobytes()
        result['compressor'][arch][name]={'cases':11,'capture_controls_exact':3,'records':records}
    outputs={name:load(R/f'attention-{name}-{arch}/w1.f32') for name in ['w1','w2','w4','w2-swap']}
    fa={}
    for name,out in outputs.items():
        capture=load(R/'attention-inputs'/('w2' if name=='w2-swap' else name)/'out.f32')
        fa[name]={'vs_capture':compare(capture,out),'vs_scalar_kv':compare(outputs['w1'],out)}
        if arch=='ada':assert fa[name]['vs_capture']['bit_identical']
    assert outputs['w2'].tobytes()==outputs['w4'].tobytes()==outputs['w2-swap'].tobytes()
    result['attention'][arch]={'cases':4,'row1596_swap_matches_wide_bit_identical':True,'records':fa}
    for name in ['w2','w4','w2-swap']:
        for f in ['q.f32','mask.f16','sinks.f32']:
            assert (R/'attention-inputs'/name/f).read_bytes()==(R/'attention-inputs'/'w1'/f).read_bytes()
a=(R/'attention-inputs/w1/k.f16').read_bytes();b=(R/'attention-inputs/w2/k.f16').read_bytes()
mask=(R/'attention-inputs/w1/mask.f16').read_bytes()
diff=[]
for row in range(1792):
    if not math.isfinite(struct.unpack_from('<e',mask,row*2)[0]):continue
    for col in range(512):
        at=(row*512+col)*2
        if a[at:at+2]!=b[at:at+2]:diff.append({'row':row,'component':col,'scalar':struct.unpack_from('<e',a,at)[0],'wide':struct.unpack_from('<e',b,at)[0]})
result['visible_kv_differences']=diff
(R/'compressor-isolated-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
