#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
metric = ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

def read(p):
    a = array('f')
    a.frombytes(p.read_bytes())
    return a

def compare(a, b):
    o = (ctypes.c_double*5)()
    assert len(a) == len(b)
    metric(a.buffer_info()[0], b.buffer_info()[0], len(a), 0, o)
    return {'max_abs': o[2], 'rms': math.sqrt(o[0]/len(a)), 'nonfinite': int(o[4]), 'bit_identical': a.tobytes() == b.tobytes()}

f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
f16 = lambda x: struct.unpack('<e', struct.pack('<e', x))[0]

def q8_model(a):
    q, scales = [], []
    for start in range(0, len(a), 32):
        group = a[start:start+32]
        d = f32(max(abs(x) for x in group)/127)
        scales.append(f16(d))
        for x in group:
            v = f32(x/d) if d else 0
            q.append(math.floor(v+.5) if v >= 0 else math.ceil(v-.5))
    return q, scales

summary = {'note': 'Fixed-input Q6_K replay isolates batch kernel effects; FP32 reuses the same dequantized weights, not a higher-quality checkpoint', 'variants': {}}
rounding = {'note': 'CPU model of CUDA quantize_q8_1 equations, not a GPU integer capture; Q8 CPU projection agreement is checked independently', 'variants': {}}
for kind in ['projection', 'projection-qa']:
    inputs = {w: read(ROOT/f'{kind}-inputs/input-w{w}.f32') for w in [1, 2, 4]}
    quant = {w: q8_model(x) for w, x in inputs.items()}
    per = {'input_difference': {}, 'capture_reproduction': {}, 'batch_controls': {}, 'output_sensitivity': {}, 'q8_cpu_reference': {}}
    per_rounding = {}
    for width in [2, 4]:
        per['input_difference'][str(width)] = compare(inputs[1], inputs[width])
        q1, d1 = quant[1]
        qw, dw = quant[width]
        per_rounding[str(width)] = {'changed_q8_indices': [i for i, (a, b) in enumerate(zip(q1, qw)) if a != b],
            'changed_half_scale_blocks': [i for i, (a, b) in enumerate(zip(d1, dw)) if a != b]}
    for arch in ['ada', 'ampere']:
        directory = ROOT/(kind+'-'+arch)
        per['batch_controls'][arch] = [json.loads(l) for l in (directory/'results.jsonl').read_text().splitlines()]
        for width in [1, 2, 4]:
            captured = read(ROOT/f'{kind}-inputs/captured-w{width}.f32')
            actual = read(directory/f'quant-inputw{width}-batch{width}.f32')[:len(captured)]
            per['capture_reproduction'][f'{arch}-w{width}'] = compare(captured, actual)
            q8 = read(ROOT/f'{kind}-q8-reference-output/inputw{width}.f32')
            per['q8_cpu_reference'][f'{arch}-w{width}'] = compare(q8, actual)
        for mode in ['quant', 'fp32', 'cpu']:
            paths = [directory/(f'cpu-inputw{w}.f32' if mode == 'cpu' else f'{mode}-inputw{w}-batch1.f32') for w in [1, 2, 4]]
            values = [read(p) for p in paths]
            for width, x in zip([2, 4], values[1:]):
                per['output_sensitivity'][f'{arch}-{mode}-w{width}'] = compare(values[0], x)
    summary['variants'][kind] = per
    rounding['variants'][kind] = per_rounding
(ROOT/'projection-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(ROOT/'q8-input-rounding.json').write_text(json.dumps(rounding, indent=2)+'\n')
assert all(v['bit_identical'] for p in summary['variants'].values() for k, v in p['capture_reproduction'].items() if k.startswith('ada-'))
assert all(v['max_vs_scalar'] == 0 and v['column_max'] == 0 and v['repeat_stable'] for p in summary['variants'].values() for rows in p['batch_controls'].values() for v in rows if v['label'].startswith('quant-'))
print(json.dumps({k: {'input_difference': v['input_difference'], 'output_sensitivity': v['output_sensitivity'], 'capture_reproduction': v['capture_reproduction']} for k,v in summary['variants'].items()}, indent=2))
