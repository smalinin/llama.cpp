import json,math,runpy
from pathlib import Path

R=Path(__file__).resolve().parent
helper=runpy.run_path(str(R/'analyze-attention.py'))
read_tensor=helper['read_tensor'];compare=helper['compare'];AXES=helper['AXES']
MODE='decode-scalar-fa-upgate-hc-router-down'
D=R/'compressor-candidate-output'
def capture(mode,w,t):
    start=t//w*w;d=D/f'{mode}-w{w}-input{start}'
    ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
    return d,{(a['owner'],a['role']):a for a in ts if a['owner']!='comp_kv-2'},t-start

result={}
for suffix in ['-compressor','-float2d']:
    records=[]
    for t in range(20):
        da,ta,ca=capture(MODE,1,t)
        for w in [2,4]:
            db,tb,cb=capture(MODE+suffix,w,t)
            item={'input_index':t,'width':w,'tensors':{}}
            for key in ta:
                if key in [('FA-2','src1'),('FA-2','src2'),('FA-2','src3')]:continue
                axis=AXES.get(key,1)
                if key==('FA-2','src4'):axis=None
                a=read_tensor(da,ta[key],axis,ca if axis is not None else 0)
                b=read_tensor(db,tb[key],axis,cb if axis is not None else 0)
                item['tensors']['/'.join(key)]=compare(a,b)
            ma=read_tensor(da,ta[('FA-2','src3')],1,ca);mb=read_tensor(db,tb[('FA-2','src3')],1,cb)
            assert ma.tobytes()==mb.tobytes()
            a=(da/ta[('FA-2','src1')]['file']).read_bytes();b=(db/tb[('FA-2','src1')]['file']).read_bytes()
            visible=[i for i,v in enumerate(ma) if math.isfinite(v)]
            assert all(a[i*1024:(i+1)*1024]==b[i*1024:(i+1)*1024] for i in visible)
            for key,v in item['tensors'].items():
                if not key.startswith('idx_top_k-2/'):assert v['bit_identical'],(suffix,t,w,key,v)
            item['mask_and_visible_kv_bit_identical']=True
            records.append(item)
    result[suffix]={'cases':40,'comparisons':records,'all_attention_outputs_bit_identical':True,
                    'note':'Indexer order may differ with the same visible mask; masks and all visible K/V rows are checked separately.'}
(R/'candidate-boundaries-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{s:v[s] for s in ['cases','all_attention_outputs_bit_identical']} for k,v in result.items()},indent=2))
