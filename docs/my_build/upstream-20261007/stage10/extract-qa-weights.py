#!/usr/bin/env python3
import ctypes
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
model = Path(json.loads((ROOT.parent / 'stage7/model-inspection.json').read_text())['models']['model']['path'])
output = ROOT/'projection-qa-inputs'
output.mkdir(exist_ok=True)
lib = ctypes.CDLL(str(ROOT.parent/'stage8/candidate-bin/libggml-base.so'))
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
lib.ggml_row_size.restype = ctypes.c_size_t
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
            if name not in ('blk.3.attn_q_a.weight','blk.3.attn_q_a_norm.weight'):
                continue
            f.seek(start + offset)
            size = lib.ggml_row_size(kind, dims[0])*math.prod(dims[1:])
            raw = f.read(size)
            assert len(raw) == size
            (output/('norm.f32' if name.endswith('_norm.weight') else 'weight.bin')).write_bytes(raw)
            manifest = {'source': str(path), 'tensor': name, 'shape': dims, 'type': kind,
                        'offset': start + offset, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            (output / (name + '-extraction.json')).write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(manifest))
