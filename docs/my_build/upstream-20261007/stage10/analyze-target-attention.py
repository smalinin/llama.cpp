#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT/'target-attention-output'
METRIC = ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
METRIC.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
V = 129280

def read(p):
    x = array('f')
    with p.open('rb') as f:
        x.fromfile(f, p.stat().st_size//4)
    return x

def compare(a, b):
    assert len(a) == len(b)
    o = (ctypes.c_double*5)()
    METRIC(a.buffer_info()[0], b.buffer_info()[0], len(a), 0, o)
    return {'max_abs': o[2], 'rms': math.sqrt(o[0]/len(a)),
            'reference_rms': math.sqrt(o[1]/len(a)), 'nonfinite': int(o[4]),
            'bit_identical': a.tobytes() == b.tobytes()}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

summary = {'rows': 65, 'forced_history': 'Stage9 current native first 64 tokens', 'controls': {}, 'variants': {}, 'layer_features': {}}
for width in [1, 2, 4]:
    p = D/f'barrier-w{width}-logits.f32'
    old = ROOT.parent/f'stage9/controls-output/default-w{width}-logits.f32'
    summary['controls'][f'barrier-w{width}'] = {'sha256': sha(p), 'stage9_sha256': sha(old), 'bit_identical': sha(p) == sha(old)}
    p = D/f'decode-features-w{width}-logits.f32'
    old = ROOT.parent/f'stage9/attention-capture-output/decode-w{width}-logits.f32'
    summary['controls'][f'decode-w{width}'] = {'sha256': sha(p), 'stage9_sha256': sha(old), 'bit_identical': sha(p) == sha(old)}
for mode in ['barrier', 'decode-features', 'scalar-fa', 'decode-scalar-fa-features']:
    base = read(D/f'{mode}-w1-logits.f32')
    rows1 = [json.loads(l) for l in (D/f'{mode}-w1-rows.jsonl').read_text().splitlines()]
    for width in [1, 2, 4]:
        label = f'{mode}-w{width}'
        x = read(D/f'{label}-logits.f32')
        rows = [json.loads(l) for l in (D/f'{label}-rows.jsonl').read_text().splitlines()]
        result = compare(base, x)
        result.update(sha256=sha(D/f'{label}-logits.f32'),
            argmax_difference_indices=[i for i, (a, b) in enumerate(zip(rows1, rows)) if a['argmax'] != b['argmax']],
            width1_difference_vs_forced=[i for i, a in enumerate(rows1) if a['argmax'] != a['reference']],
            index19_verify_minus_check=x[19*V+23393]-x[19*V+4085],
            row1=compare(base[V:2*V], x[V:2*V]), row19=compare(base[19*V:20*V], x[19*V:20*V]))
        summary['variants'][label] = result
    if mode.endswith('-features'):
        base = read(D/f'{mode}-w1-features.f32')
        for width in [2, 4]:
            x = read(D/f'{mode}-w{width}-features.f32')
            assert len(x) == len(base) == 65*41*5120
            per = []
            for row in [1, 17, 19, 33]:
                for layer in range(41):
                    start = (row*41+layer)*5120
                    m = compare(base[start:start+5120], x[start:start+5120])
                    m.update(row=row, layer=layer)
                    per.append(m)
            summary['layer_features'][f'{mode}-w{width}'] = per
summary['controls']['scalar-fa-w1'] = {'bit_identical': summary['variants']['scalar-fa-w1']['sha256'] == summary['variants']['barrier-w1']['sha256']}
summary['controls']['decode-scalar-fa-w1'] = {'bit_identical': summary['variants']['decode-scalar-fa-features-w1']['sha256'] == summary['variants']['decode-features-w1']['sha256']}
(D/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
assert all(v['bit_identical'] for v in summary['controls'].values()), summary['controls']
print(json.dumps({'controls': summary['controls'], 'variants': summary['variants']}, indent=2))
