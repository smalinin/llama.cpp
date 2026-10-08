#!/usr/bin/env python3
from array import array
import ctypes
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
ref = ctypes.CDLL(str(ROOT.parent/'stage9/attention-reference.so')).attention_reference
ref.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]*4+[ctypes.c_void_p]

def read(name):
    if name.endswith('.f16'):
        return array('f', (v[0] for v in struct.iter_unpack('<e', (ROOT/name).read_bytes())))
    x = array('f')
    with (ROOT/name).open('rb') as f:
        x.fromfile(f, (ROOT/name).stat().st_size//4)
    return x

inputs = [read(p) for p in ['q.f32','k.f16','v.f16','mask.f16','sinks.f32']]
metric = ctypes.CDLL(str(ROOT.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
summary = {'reference': 'CPU double reduction; does not emulate CUDA online softmax or F16 probability tiles', 'variants': {}}
for rounded in [0, 1]:
    output = array('f', [0])*(512*64)
    ref(*(x.buffer_info()[0] for x in inputs), 1792, 64, 512, rounded, output.buffer_info()[0])
    for arch in ['ada', 'ampere']:
        for width in [1, 2, 3, 4, 5, 8, 16]:
            actual = read(f'attention-{arch}/w{width}.f32')[:512*64]
            o = (ctypes.c_double*5)()
            metric(output.buffer_info()[0], actual.buffer_info()[0], len(output), 0, o)
            summary['variants'][f'{arch}-w{width}-rounded{rounded}'] = {
                'max_abs': o[2], 'rms': (o[0]/len(output))**.5, 'nonfinite': int(o[4])}
(ROOT/'attention-reference-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
