#!/usr/bin/env python3
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'capture-base'

def index(directory):
    counts = collections.Counter()
    result = {}
    for line in (directory / 'tensors.jsonl').read_text().splitlines():
        tensor = json.loads(line)
        key = (tensor['name'], tensor['op'], counts[tensor['name'], tensor['op']])
        counts[tensor['name'], tensor['op']] += 1
        result[key] = tensor
    return result

def values(directory, tensor, shape):
    raw = (directory / tensor['file']).read_bytes()
    fmt = {0: 'f', 1: 'e', 26: 'i', 27: 'q'}[tensor['type']]
    unpack = struct.Struct('<' + fmt).unpack_from
    nb = tensor['nb']
    return [unpack(raw, sum(p * stride for p, stride in zip(coords[::-1], nb)))[0]
            for coords in itertools.product(*(range(n) for n in shape[::-1]))]

reference = index(DATA / 'plain-w1')
results = {}
for label in ('plain-w2', 'plain-w4', 'prefix-w1-switch-w2', 'prefix-w1-switch-w4'):
    compared = index(DATA / label)
    rows = []
    for key, left in reference.items():
        right = compared.get(key)
        if right is None:
            rows.append({'name': key[0], 'op': key[1], 'status': 'missing'})
            continue
        a, b = left['ne'], right['ne']
        shape = list(a)
        axes = [i for i in range(4) if a[i] != b[i]]
        if axes and (len(axes) != 1 or a[axes[0]] != 1):
            rows.append({'name': key[0], 'op': key[1], 'status': 'shape-skipped', 'left': a, 'right': b})
            continue
        x, y = values(DATA / 'plain-w1', left, shape), values(DATA / label, right, shape)
        delta = [abs(i-j) for i, j in zip(x, y)]
        rms = math.sqrt(sum(d*d for d in delta) / len(delta))
        xrms = math.sqrt(sum(float(v)*float(v) for v in x) / len(x))
        rows.append({'name': key[0], 'op': key[1], 'status': 'compared', 'count': len(x),
                     'token_axis': axes[0] if axes else None,
                     'max_abs': max(delta), 'rms': rms, 'reference_rms': xrms,
                     'different': sum(d != 0 for d in delta), 'file': left['file'],
                     'compared_file': right['file']})
    results[label] = rows

checks = []
for width in (1, 2, 4):
    name = f'plain-w{width}-logits.f32'
    a = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    b = hashlib.sha256((ROOT.parent / 'stage7/target-replay-server-matched-output' / name).read_bytes()).hexdigest()
    checks.append({'width': width, 'sha256': a, 'stage7_sha256': b, 'identical': a == b})
(ROOT / 'capture-analysis.json').write_text(json.dumps({'logits_checks': checks, 'comparisons': results}, indent=2) + '\n')
print('Logits checks:', checks)
for label, rows in results.items():
    print(label)
    for row in rows:
        if row['status'] == 'compared' and (row['name'].endswith('-0') or row['name'].startswith(('l_last-', 'engram_'))):
            print(row['name'], row['op'], row['max_abs'], row['rms'], row['reference_rms'])
