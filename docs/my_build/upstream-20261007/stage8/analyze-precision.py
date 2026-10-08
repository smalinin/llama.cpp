#!/usr/bin/env python3
from array import array
import ctypes
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
directory = Path(sys.argv[1])
metrics = ctypes.CDLL(str(ROOT.parent / 'stage5/metrics.so'))
metrics.stage5_metrics.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_double)]

def read(label):
    path = directory / (label + '-logits.f32')
    result = array('f')
    with path.open('rb') as f:
        result.fromfile(f, path.stat().st_size // 4)
    assert len(result) == 65*129280
    return result

base = read('plain-w1')
reference = [json.loads(line) for line in (directory / 'plain-w1-rows.jsonl').read_text().splitlines()]
summary = {'rows': 65, 'vocab': 129280,
           'forced_prefix': 'Stage7 native tokens; not a native-output reference for the modified build',
           'width1_argmax_differs_from_forced_at': [i for i, row in enumerate(reference) if row['argmax'] != row['reference']],
           'variants': {}}
for path in sorted(directory.glob('*-rows.jsonl')):
    label = path.name.removesuffix('-rows.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    values = read(label)
    out = (ctypes.c_double * 5)()
    metrics.stage5_metrics(base.buffer_info()[0], values.buffer_info()[0], len(base), 0, out)
    assert out[4] == 0
    summary['variants'][label] = {
        'argmax_difference_indices_vs_plain_w1': [i for i, pair in enumerate(zip(reference, rows)) if pair[0]['argmax'] != pair[1]['argmax']],
        'max_abs_logits_vs_plain_w1': out[2], 'rms_logits_vs_plain_w1': math.sqrt(out[0]/len(base)),
        'identical_to_plain_w1': values.tobytes() == base.tobytes(),
        'sha256': hashlib.sha256(values).hexdigest(),
        'index33': rows[33], 'index33_entries_minus_checkpoint': values[33*129280+23914]-values[33*129280+72888]}
(directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
