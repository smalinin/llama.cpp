#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('inspection', ROOT.parent / 'stage7/inspect-models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def table(path):
    with path.open('rb') as f:
        magic, version, count, keys = struct.unpack('<4sIQQ', f.read(24))
        assert magic == b'GGUF' and version == 3
        metadata = {}
        for _ in range(keys):
            key = module.string(f)
            metadata[key] = module.value(f, module.u32(f))
        tensors = []
        for _ in range(count):
            name = module.string(f)
            dims = [module.u64(f) for _ in range(module.u32(f))]
            kind, offset = module.u32(f), module.u64(f)
            block, size = {0: (1, 4), 8: (32, 34), 12: (256, 144), 30: (1, 2), 39: (32, 17)}[kind]
            tensors.append({'name': name, 'type': kind, 'offset': offset, 'bytes': math.prod(dims)//block*size})
        alignment = metadata.get('general.alignment', 32)
        start = (f.tell()+alignment-1)//alignment*alignment
        return {t['name']: dict(t, offset=t['offset']+start) for t in tensors}

def digest(f, t):
    f.seek(t['offset'])
    count, h = t['bytes'], hashlib.sha256()
    while count:
        raw = f.read(min(count, 8*1024*1024))
        assert raw
        h.update(raw)
        count -= len(raw)
    return h.hexdigest()

models = json.loads((ROOT / 'q4-comparison-manifest.json').read_text())['models']
paths = {key: Path(value['path']) for key, value in models.items()}
tables = {key: table(path) for key, path in paths.items()}
checks = []
with paths['mxfp4'].open('rb') as a, paths['q4k'].open('rb') as b:
    for name, t in tables['mxfp4'].items():
        if t['type'] == 39:
            continue
        other = tables['q4k'][name]
        assert t['bytes'] == other['bytes'] and t['type'] == other['type']
        x, y = digest(a, t), digest(b, other)
        assert x == y, name
        checks.append({'tensor': name, 'bytes': t['bytes'], 'sha256': x, 'identical': True})
assert len(checks) == 69
(ROOT / 'q4-preserved-tensors.json').write_text(json.dumps(checks, indent=2) + '\n')
print('All 69 unchanged tensors are bit-identical')
