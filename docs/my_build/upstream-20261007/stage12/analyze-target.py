#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
D=ROOT/'target-ffn-output'
OLD=ROOT.parent/'stage10/target-attention-output'
metric=ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
V=129280
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read(p,n=None):
    a=array('f');a.frombytes(p.read_bytes())
    return a if n is None else a[:n]

def compare(a,b):
    assert len(a)==len(b) and a
    o=(ctypes.c_double*5)()
    metric(a.buffer_info()[0],b.buffer_info()[0],len(a),0,o)
    assert o[4]==0
    return {'max_abs':o[2],'rms':math.sqrt(o[0]/len(a)),'bit_identical':a.tobytes()==b.tobytes()}

summary={'rows':65,'forced_history':'Stage9 current native first64 tokens; no draft or rollback',
         'controls':{},'variants':{},'first_layer2_hidden':{},'replaced_queries':{}}
for w in [1,2,4]:
    for mode,oldmode in [('barrier','barrier'),('decode-scalar-fa','decode-scalar-fa-features')]:
        p=D/f'{mode}-w{w}-logits.f32'
        old=OLD/f'{oldmode}-w{w}-logits.f32'
        summary['controls'][f'{mode}-w{w}']={'sha256':sha(p),'reference_sha256':sha(old),'bit_identical':sha(p)==sha(old)}
for mode in ['barrier','scalar-upgate','decode-scalar-fa','decode-scalar-fa-upgate']:
    base=read(D/f'{mode}-w1-logits.f32')
    assert len(base)==65*V
    rows1=[json.loads(l) for l in (D/f'{mode}-w1-rows.jsonl').read_text().splitlines()]
    assert len(rows1)==65 and all(r['argmax']==r['reference'] for r in rows1)
    if mode.endswith('upgate'):
        p=D/f'{mode}-w1-logits.f32'
        summary['controls'][mode+'-scalar']={'sha256':sha(p),'bit_identical':sha(p)==sha(D/'barrier-w1-logits.f32')}
    for w in [1,2,4]:
        label=f'{mode}-w{w}'
        p=D/(label+'-logits.f32'); x=read(p)
        assert len(x)==len(base)
        rows=[json.loads(l) for l in (D/(label+'-rows.jsonl')).read_text().splitlines()]
        assert len(rows)==65
        result=compare(base,x)
        result.update(sha256=sha(p),argmax_difference_indices=[i for i,(a,b) in enumerate(zip(rows1,rows)) if a['argmax']!=b['argmax']],
                      index19_verify_minus_check=x[19*V+23393]-x[19*V+4085])
        summary['variants'][label]=result
        before=read(D/(label+'-ffn_moe_swiglu_limited-2-before.f32'),2304*6)
        after=read(D/(label+'-ffn_moe_swiglu_limited-2-after.f32'),2304*6)
        native=read(D/(f'{mode}-w1-ffn_moe_swiglu_limited-2-after.f32'))
        summary['first_layer2_hidden'][label]={'before_vs_scalar':compare(before,native),'after_vs_scalar':compare(after,native)}
        if 'upgate' not in mode:assert before.tobytes()==after.tobytes()
        if mode=='decode-scalar-fa':
            name=f'inputw{w}-batch{w}-'+('repeat' if w==1 else 'actual')+'-moe-hidden.f32'
            old=ROOT.parent/'stage11/ffn-ada'/name
            assert (D/(label+'-ffn_moe_swiglu_limited-2-before.f32')).read_bytes()==old.read_bytes()
assert all(c['bit_identical'] for c in summary['controls'].values())
assert all(summary['first_layer2_hidden'][f'decode-scalar-fa-upgate-w{w}']['after_vs_scalar']['bit_identical'] for w in [2,4])
log=(ROOT/'target-ffn.log').read_text(errors='replace')
for label,count in re.findall(r'REPLACED_UPGATE (\S+) (\d+) token-layers',log):
    summary['replaced_queries'][label]=int(count)
assert len(summary['replaced_queries'])==12
assert all(count==(2560 if 'upgate' in label and not label.endswith('w1') else 0) for label,count in summary['replaced_queries'].items())
summary['conclusion']='Neither scalar up/gate alone nor combined with FP32 floating projections and scalar attention restores general batched greedy compatibility. Width2 combined matches 65 argmax values; width4 still differs at index19. Free-generation DSpark was not rerun for this rejected candidate.'
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'controls_exact':len(summary['controls']),'variants':{k:v for k,v in summary['variants'].items() if not k.endswith('w1')}},indent=2))
