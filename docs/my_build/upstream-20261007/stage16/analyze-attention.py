import hashlib,json,math,runpy
from pathlib import Path

R=Path(__file__).resolve().parent
helpers=runpy.run_path(str(R/'analyze-chain.py'))
read_tensor=helpers['read_tensor'];compare=helpers['compare']
D=R/'attention-capture-output'
MODE='decode-scalar-fa-upgate-hc-router-down'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
AXES={('q-2','output'):2,('kv-2','output'):2,('FA-2','output'):2,
      ('FA-2','src0'):1,('FA-2','src3'):1,('attn_derope-2','output'):2}

def capture(w,t):
    start=t//w*w
    d=D/f'{MODE}-w{w}-input{start}'
    ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
    items={(a['owner'],a['role']):a for a in ts if a['owner']!='comp_kv-2'}
    return d,items,t-start

def values(d,ts,col,key):
    a=ts[key]
    axis=AXES.get(key,1)
    if key==('FA-2','src4'):axis=None
    return read_tensor(d,a,axis,col if axis is not None else 0)

if __name__=='__main__':
    result={'controls':{},'comparisons':[]}
    for w in [1,2,4]:
        p=D/f'{MODE}-w{w}-logits.f32'
        old=R.parent/'stage15/down-candidate-output'/p.name
        assert sha(p)==sha(old)
        result['controls'][str(w)]=sha(p)
    for t in range(20):
        d1,ts1,c1=capture(1,t)
        for w in [2,4]:
            dw,tsw,cw=capture(w,t)
            item={'input_index':t,'output_index':t+1,'width':w,'batch_start':t//w*w,'column':cw,'tensors':{}}
            for key in ts1:
                if key in [('FA-2','src1'),('FA-2','src2'),('FA-2','src3')]:continue
                a=values(d1,ts1,c1,key);b=values(dw,tsw,cw,key)
                item['tensors']['/'.join(key)]=compare(a,b)
            m1=values(d1,ts1,c1,('FA-2','src3'));mw=values(dw,tsw,cw,('FA-2','src3'))
            assert len(m1)==len(mw)
            visible=[i for i,v in enumerate(m1) if math.isfinite(v)]
            item['mask']={'bit_identical':m1.tobytes()==mw.tobytes(),'visible':len(visible),
                          'visible_sets_equal':visible==[i for i,v in enumerate(mw) if math.isfinite(v)]}
            a=ts1[('FA-2','src1')];b=tsw[('FA-2','src1')]
            ra=(d1/a['file']).read_bytes();rb=(dw/b['file']).read_bytes()
            assert a['ne']==b['ne'] and a['nb']==b['nb']
            step=a['nb'][1]
            diffs=[i for i in visible if ra[i*step:(i+1)*step]!=rb[i*step:(i+1)*step]]
            item['visible_kv']={'bit_identical':not diffs,'different_rows':diffs}
            if diffs:
                av=read_tensor(d1,a);bv=read_tensor(dw,b)
                from array import array
                item['visible_kv']['metrics']=compare(array('f',(av[i*512+j] for i in diffs for j in range(512))),
                                                     array('f',(bv[i*512+j] for i in diffs for j in range(512))))
            for dc,tc in [(d1,ts1),(dw,tsw)]:
                assert (dc/tc[('FA-2','src1')]['file']).read_bytes()==(dc/tc[('FA-2','src2')]['file']).read_bytes()
            result['comparisons'].append(item)
    (R/'attention-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for w in [2,4]:
        rows=[r for r in result['comparisons'] if r['width']==w]
        print('WIDTH',w)
        for key in rows[0]['tensors']:
            first=next((r for r in rows if not r['tensors'][key]['bit_identical']),None)
            print(key, None if first is None else (first['input_index'],first['tensors'][key]))
        for r in rows[:4]:print(r['input_index'],r['mask'],r['visible_kv'])
