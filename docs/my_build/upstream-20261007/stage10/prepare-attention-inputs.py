#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT.parent/'stage9/attention-capture-output/plain-w1'
tensors = [json.loads(line) for line in (source/'tensors.jsonl').read_text().splitlines()]
manifest = {'source': 'stage9/attention-capture-output/plain-w1', 'D': 512, 'heads': 64, 'n_kv': 1792, 'files': {}}
for name, index, suffix in [('q', 6, 'f32'), ('k', 7, 'f16'), ('v', 8, 'f16'),
                             ('mask', 9, 'f16'), ('sinks', 10, 'f32'), ('out', 5, 'f32')]:
    tensor = tensors[index]
    data = (source/tensor['file']).read_bytes()
    element_size = 4 if suffix == 'f32' else 2
    dense = bytearray()
    for i3 in range(tensor['ne'][3]):
        for i2 in range(tensor['ne'][2]):
            for i1 in range(tensor['ne'][1]):
                for i0 in range(tensor['ne'][0]):
                    offset = sum(i*b for i, b in zip([i0, i1, i2, i3], tensor['nb']))
                    dense.extend(data[offset:offset+element_size])
    output = ROOT/(name+'.'+suffix)
    if output.exists():
        assert output.read_bytes() == dense, output
    else:
        output.write_bytes(dense)
    manifest['files'][name] = {'name': output.name, 'bytes': len(dense),
        'sha256': hashlib.sha256(dense).hexdigest(), 'source_tensor': tensor}
(ROOT/'attention-input-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
