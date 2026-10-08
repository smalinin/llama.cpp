#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct
ROOT=Path(__file__).resolve().parent
D=ROOT/'attention-capture-output'
lib=ctypes.CDLL(str(ROOT/'attention-reference.so'))
ref=lib.attention_reference
ref.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*4+[ctypes.c_void_p]
proj=lib.bf16_projection_reference
proj.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

def values(directory,tensor):
    raw=(directory/tensor['file']).read_bytes()
    code={0:'f',1:'e'}[tensor['type']]
    size={0:4,1:2}[tensor['type']]
    ne,nb=tensor['ne'],tensor['nb']
    def at(i0=0,i1=0,i2=0,i3=0):
        return struct.unpack_from('<'+code,raw,i0*nb[0]+i1*nb[1]+i2*nb[2]+i3*nb[3])[0]
    return at

def compare(a,b):
    assert len(a)==len(b)
    diff=[float(x)-float(y) for x,y in zip(a,b)]
    return {'count':len(a),'max_abs':max(map(abs,diff),default=0),
            'rms':math.sqrt(sum(x*x for x in diff)/max(1,len(diff))),
            'reference_rms':math.sqrt(sum(float(x)*float(x) for x in a)/max(1,len(a))),
            'different':sum(x!=y for x,y in zip(a,b))}

summary={'reference':'CPU FP64 attention reductions; optional Q and scale half rounding. CUDA online softmax and half probability tiles are not emulated.', 'instrumentation_logits_integrity':{},'variants':{}}
features={}
for width in [1,2,4]:
    directory=D/f'plain-w{width}'
    rows=[json.loads(s) for s in (directory/'tensors.jsonl').read_text().splitlines()]
    fidx=next(i for i,x in enumerate(rows) if x['op']=='FLASH_ATTN_EXT')
    fa=rows[fidx]
    q,k,v,mask,sinks=rows[fidx+1:fidx+6]
    q_at,k_at,v_at,m_at,s_at=[values(directory,x) for x in [q,k,v,mask,sinks]]
    nk=k['ne'][1];nh=q['ne'][2];d=q['ne'][0]
    query=array('f',[q_at(j,0,h) for h in range(nh) for j in range(d)])
    keys=array('f',[k_at(j,i) for i in range(nk) for j in range(d)])
    vals=array('f',[v_at(j,i) for i in range(nk) for j in range(d)])
    masks=array('f',[m_at(i,0) for i in range(nk)])
    sink=array('f',[s_at(h) for h in range(nh)])
    out_at=values(directory,fa)
    actual=array('f',[out_at(j,h,0) for h in range(nh) for j in range(d)])
    metrics={}
    for rounded in [0,1]:
        out=array('f',[0.0]*(nh*d))
        ref(*(x.buffer_info()[0] for x in [query,keys,vals,masks,sink]),nk,nh,d,rounded,out.buffer_info()[0])
        metrics['cpu_double_q_scale_half' if rounded else 'cpu_double_q_float']=compare(out,actual)
    metrics.update(n_kv=nk,mask_finite=sum(math.isfinite(x) for x in masks),
                   mask_allowed=[i for i,x in enumerate(masks) if math.isfinite(x)],projections={})
    features[width]=(query,keys,vals,masks,actual,sink)
    for name,weight in [('comp_state_kv-2','blk.2.attn_compressor_kv.weight'),('comp_state_score-2','blk.2.attn_compressor_gate.weight')]:
        idx=next(i for i,x in enumerate(rows) if x['name']==name)
        tensor,input_tensor=rows[idx:idx+2]
        output_at,input_at=values(directory,tensor),values(directory,input_tensor)
        x=array('f',[input_at(j,0) for j in range(5120)])
        actual_projection=[output_at(i,0) for i in range(512)]
        w=ctypes.create_string_buffer((ROOT/(weight+'.bf16')).read_bytes())
        modes={}
        for rounded in [0,1]:
            out=array('f',[0.0]*512)
            proj(w,x.buffer_info()[0],5120,512,rounded,out.buffer_info()[0])
            modes['cpu_double_x_bf16' if rounded else 'cpu_double_x_float']=compare(out,actual_projection)
        metrics['projections'][name]=modes
    native=(ROOT/'controls-output'/f'default-w{width}-logits.f32').read_bytes()
    captured=(D/f'plain-w{width}-logits.f32').read_bytes()
    summary['instrumentation_logits_integrity'][str(width)]={'identical':native==captured,
        'baseline_sha256':hashlib.sha256(native).hexdigest(),'captured_sha256':hashlib.sha256(captured).hexdigest()}
    summary['variants'][str(width)]=metrics

for width in [2,4]:
    a=features[1];b=features[width]
    common=[i for i in range(min(len(a[3]),len(b[3]))) if math.isfinite(a[3][i]) and math.isfinite(b[3][i])]
    ka=[a[1][i*512+j] for i in common for j in range(512)]
    kb=[b[1][i*512+j] for i in common for j in range(512)]
    summary['variants'][str(width)]['vs_width1']={
        'q_first_token':compare(a[0],b[0]),'sinks':compare(a[5],b[5]),'attention_first_token':compare(a[4],b[4]),
        'visible_keys':compare(ka,kb),'common_visible_rows':len(common),
        'mask_first_token_identical':a[3].tobytes()==b[3].tobytes(),
        'visible_key_rows_differ':[i for i in common if a[1][i*512:(i+1)*512]!=b[1][i*512:(i+1)*512]]}
(ROOT/'attention-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
