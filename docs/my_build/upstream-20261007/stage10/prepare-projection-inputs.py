#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT/'projection-capture-output'
output = ROOT/'projection-inputs'
output.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'controls': {}, 'inputs': {}}
weight = None
for width in [1, 2, 4]:
    directory = D/f'decode-scalar-fa-projection-w{width}'
    tensors = [json.loads(l) for l in (directory/'tensors.jsonl').read_text().splitlines()]
    label = f'decode-scalar-fa-projection-features-w{width}'
    actual = D/(label+'-logits.f32')
    reference = ROOT/'target-attention-output'/f'decode-scalar-fa-features-w{width}-logits.f32'
    manifest['controls'][label] = {'sha256': sha(actual), 'reference_sha256': sha(reference), 'bit_identical': sha(actual) == sha(reference)}
    by_phase = {t['phase']: t for t in tensors if t['phase'].startswith('projection-')}
    w, x, y = [by_phase[p] for p in ['projection-weight', 'projection-input', 'projection-output']]
    data = (directory/w['file']).read_bytes()
    if weight is not None:
        assert weight == data
    weight = data
    (output/'weight.bin').write_bytes(data)
    manifest.update(type=w['type'], ne=w['ne'], weight_sha256=hashlib.sha256(data).hexdigest())
    for name, tensor in [('input', x), ('captured', y)]:
        data = (directory/tensor['file']).read_bytes()[:tensor['ne'][0]*4]
        p = output/f'{name}-w{width}.f32'
        p.write_bytes(data)
        manifest['inputs'][p.name] = {'source_tensor': tensor, 'sha256': sha(p)}
(output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
assert all(v['bit_identical'] for v in manifest['controls'].values())
print(json.dumps({'type': manifest['type'], 'ne': manifest['ne'], 'controls': manifest['controls']}, indent=2))
