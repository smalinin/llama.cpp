#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
D = ROOT/'layer-capture-output'
METRIC = ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
METRIC.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

def compare(a, b):
    assert len(a) == len(b)
    o = (ctypes.c_double*5)()
    METRIC(a.buffer_info()[0], b.buffer_info()[0], len(a), 0, o)
    return {'max_abs': o[2], 'rms': math.sqrt(o[0]/len(a)), 'nonfinite': int(o[4]),
            'bit_identical': a.tobytes() == b.tobytes()}

def read_tensor(directory, tensor, token_axis=None):
    data = (directory/tensor['file']).read_bytes()
    fmt = '<f' if tensor['type'] == 0 else '<e'
    ne = tensor['ne'].copy()
    if token_axis is not None:
        ne[token_axis] = 1
    result = array('f')
    for i3 in range(ne[3]):
        for i2 in range(ne[2]):
            for i1 in range(ne[1]):
                for i0 in range(ne[0]):
                    offset = sum(i*b for i, b in zip([i0, i1, i2, i3], tensor['nb']))
                    result.append(struct.unpack_from(fmt, data, offset)[0])
    return result

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {'controls': {}, 'boundaries': {}, 'attention': {}}
for mode in ['decode-scalar-fa-only', 'decode-scalar-fa-boundaries']:
    tensors = {}
    for width in [1, 2, 4]:
        label = f'{mode}-features-w{width}'
        p = D/(label+'-logits.f32')
        ref = ROOT/'target-attention-output'/f'decode-scalar-fa-features-w{width}-logits.f32'
        summary['controls'][label] = {'sha256': sha(p), 'reference_sha256': sha(ref), 'bit_identical': sha(p) == sha(ref)}
        directory = D/f'{mode}-w{width}'
        tensors[width] = [json.loads(l) for l in (directory/'tensors.jsonl').read_text().splitlines()]
    for width in [2, 4]:
        per = []
        for base, current in zip(tensors[1], tensors[width]):
            assert (base['phase'], base['name']) == (current['phase'], current['name'])
            if base['phase'] != 'boundary':
                continue
            axis = 2 if base['name'].startswith(('hc_attn_post', 'l_last')) else 1
            a = read_tensor(D/f'{mode}-w1', base, axis)
            b = read_tensor(D/f'{mode}-w{width}', current, axis)
            item = compare(a, b)
            item.update(name=base['name'], op=base['op'])
            per.append(item)
        summary['boundaries'][f'{mode}-w{width}'] = per
        for layer in [2, 3]:
            block1 = [t for t in tensors[1] if t['phase'] != 'boundary'][(layer-2)*7:(layer-1)*7]
            blockw = [t for t in tensors[width] if t['phase'] != 'boundary'][(layer-2)*7:(layer-1)*7]
            assert len(block1) == len(blockw) == 7
            a = [read_tensor(D/f'{mode}-w1', t, 1 if i in [1,4] else (2 if i in [0,6] else None)) for i, t in enumerate(block1)]
            b = [read_tensor(D/f'{mode}-w{width}', t, 1 if i in [1,4] else (2 if i in [0,6] else None)) for i, t in enumerate(blockw)]
            masks_same = a[4].tobytes() == b[4].tobytes()
            visible1 = [i for i, v in enumerate(a[4]) if math.isfinite(v)]
            visiblew = [i for i, v in enumerate(b[4]) if math.isfinite(v)]
            shared = sorted(set(visible1) & set(visiblew))
            rows = []
            for row in shared:
                x, y = a[2][row*512:(row+1)*512], b[2][row*512:(row+1)*512]
                m = compare(x, y)
                if not m['bit_identical']:
                    m['row'] = row
                    rows.append(m)
            summary['attention'][f'{mode}-w{width}-layer{layer}'] = {
                'q': compare(a[1], b[1]), 'sinks': compare(a[5], b[5]),
                'original_fa': compare(a[0], b[0]), 'scalar_fa': compare(a[6], b[6]),
                'mask_identical': masks_same, 'visible_counts': [len(visible1), len(visiblew)],
                'visibility_difference_rows': sorted(set(visible1)^set(visiblew)), 'changed_visible_k_rows': rows,
                'kv_equal_in_each_run': a[2].tobytes() == a[3].tobytes() and b[2].tobytes() == b[3].tobytes()}
(D/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
assert all(v['bit_identical'] for v in summary['controls'].values()), summary['controls']
