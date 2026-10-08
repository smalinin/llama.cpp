#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent
D=ROOT/'chain-capture-output'
MODE='decode-scalar-fa-upgate-boundaries'
metric=ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read_tensor(directory,t,axis=None):
    raw=(directory/t['file']).read_bytes();ne=t['ne'].copy()
    if axis is not None:ne[axis]=1
    a=array('f')
    for i3 in range(ne[3]):
      for i2 in range(ne[2]):
       for i1 in range(ne[1]):
        for i0 in range(ne[0]):
         offset=sum(i*b for i,b in zip([i0,i1,i2,i3],t['nb']))
         if t['type']==0:v=struct.unpack_from('<f',raw,offset)[0]
         elif t['type']==30:v=struct.unpack('<f',struct.pack('<I',struct.unpack_from('<H',raw,offset)[0]<<16))[0]
         elif t['type']==26:v=struct.unpack_from('<i',raw,offset)[0]
         else:raise ValueError(t['type'])
         a.append(v)
    return a

def compare(a,b):
    assert len(a)==len(b)
    o=(ctypes.c_double*5)();metric(a.buffer_info()[0],b.buffer_info()[0],len(a),0,o)
    assert o[4]==0
    return {'max_abs':o[2],'rms':math.sqrt(o[0]/len(a)),'bit_identical':a.tobytes()==b.tobytes()}

def axis(t):
    prefix=t['owner'].rsplit('-',1)[0]
    if t['role']!='output':
        if prefix=='hc_mixes':return None if t['role']=='src0' else 1
        return 2 if t['role'] in ['src1','src3'] else 1
    return 2 if prefix in ['hc_comb','hc_attn_post','l_last','ffn_moe_weights_scaled','ffn_moe_swiglu_limited'] else 1

if __name__=='__main__':
    tensors={w:[json.loads(l) for l in (D/f'{MODE}-w{w}/tensors.jsonl').read_text().splitlines()] for w in [1,2,4]}
    summary={'controls':{},'comparisons':{},'counts':{w:len(ts) for w,ts in tensors.items()}}
    for w in [1,2,4]:
        p=D/f'{MODE}-w{w}-logits.f32';old=ROOT.parent/f'stage12/target-ffn-output/decode-scalar-fa-upgate-w{w}-logits.f32'
        summary['controls'][str(w)]={'sha256':sha(p),'reference_sha256':sha(old),'bit_identical':sha(p)==sha(old)}
    assert all(x['bit_identical'] for x in summary['controls'].values())
    assert len(set(summary['counts'].values()))==1
    for w in [2,4]:
        values=[];occ={}
        for a,b in zip(tensors[1],tensors[w]):
            assert (a['owner'],a['role'])==(b['owner'],b['role'])
            key=(a['owner'],a['role']);index=occ.get(key,0);occ[key]=index+1
            if axis(a) is None:
                assert (D/f'{MODE}-w1'/a['file']).read_bytes()==(D/f'{MODE}-w{w}'/b['file']).read_bytes()
                result={'max_abs':0,'rms':0,'bit_identical':True}
            else:result=compare(read_tensor(D/f'{MODE}-w1',a,axis(a)),read_tensor(D/f'{MODE}-w{w}',b,axis(b)))
            result.update(owner=a['owner'],role=a['role'],occurrence=index,op=a['op'],file1=a['file'],filew=b['file'])
            values.append(result)
        summary['comparisons'][str(w)]=values
    (ROOT/'chain-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'controls':summary['controls'],'counts':summary['counts'],'first_differences':{w:[x for x in v if not x['bit_identical']][:20] for w,v in summary['comparisons'].items()}},indent=2))
