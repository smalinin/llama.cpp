#!/usr/bin/env python3
import ctypes
import hashlib
import importlib.util
import json
import math
from array import array
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('inspection', ROOT.parent / 'stage7/inspect-models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = Path(json.loads((ROOT.parent / 'stage7/model-inspection.json').read_text())['models']['model']['path'])
output = ROOT/'inputs'
output.mkdir(exist_ok=True)
lib = ctypes.CDLL(str(ROOT.parent/'stage8/candidate-bin/libggml-base.so'))
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
lib.ggml_row_size.restype = ctypes.c_size_t
selected=[]
for path in sorted(model.parent.glob('*.gguf')):
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
            tensors.append((name, dims, kind, offset))
        alignment = metadata.get('general.alignment', 32)
        start = (f.tell() + alignment - 1) // alignment * alignment
        for name, dims, kind, offset in tensors:
            if not name.startswith(('blk.2.ffn_', 'blk.2.exp_probs_b')):
                continue
            f.seek(start + offset)
            size = lib.ggml_row_size(kind, dims[0])*math.prod(dims[1:])
            sha = hashlib.sha256()
            remaining = size
            while remaining:
                raw = f.read(min(remaining, 64*1024*1024))
                assert raw
                sha.update(raw)
                remaining -= len(raw)
            manifest = {'source': str(path), 'tensor': name, 'shape': dims, 'type': kind,
                        'offset': start + offset, 'bytes': size, 'sha256': sha.hexdigest(), 'source_size': path.stat().st_size, 'source_mtime_ns': path.stat().st_mtime_ns}
            selected.append(manifest)
            print(json.dumps(manifest))

(output/'weight-index.json').write_text(json.dumps(selected, indent=2)+'\n')
with (output/'weight-index.tsv').open('w') as f:
    for t in selected:
        dims = t['shape']+[1]*(4-len(t['shape']))
        f.write('\t'.join(map(str,[t['tensor'], t['type'], *dims, t['offset'], t['bytes'], t['source']]))+'\n')
manifest = {'input_source':'Stage10 decode-scalar-fa-boundaries, first decode call, layer2', 'files':{}}
for width in [1,2,4]:
    directory = ROOT.parent/'stage10/layer-capture-output'/f'decode-scalar-fa-boundaries-w{width}'
    tensors = [json.loads(l) for l in (directory/'tensors.jsonl').read_text().splitlines()]
    for prefix, name in [('input','ffn_norm-2'),('captured','ffn_out-2')]:
        t = next(t for t in tensors if t['name']==name)
        raw = (directory/t['file']).read_bytes()
        file = output/f'{prefix}-w{width}.f32'
        file.write_bytes(raw)
        manifest['files'][file.name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw), 'source_tensor':t}
manifest.update(experts=384,experts_used=6,weights_scale=1.5,swiglu_clamp=10.0)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
