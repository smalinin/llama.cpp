#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
output = ROOT/'projection-qa-inputs'
manifest = json.loads((output/'blk.3.attn_q_a.weight-extraction.json').read_text())
manifest['ne'] = manifest['shape']
manifest['inputs'] = {}
for width in [1, 2, 4]:
    directory = ROOT/'layer-capture-output'/f'decode-scalar-fa-boundaries-w{width}'
    tensors = [json.loads(l) for l in (directory/'tensors.jsonl').read_text().splitlines()]
    tensor = next(t for t in tensors if t['name'] == 'attn_norm-3' and t['phase'] == 'boundary')
    raw = (directory/tensor['file']).read_bytes()[:5120*4]
    path = output/f'input-w{width}.f32'
    path.write_bytes(raw)
    manifest['inputs'][path.name] = {'source_tensor': tensor, 'source': str(directory), 'sha256': hashlib.sha256(raw).hexdigest()}
    raw = (ROOT/'projection-inputs'/f'input-w{width}.f32').read_bytes()
    (output/f'captured-w{width}.f32').write_bytes(raw)
epsilon = json.loads((ROOT.parent/'stage7/model-inspection.json').read_text())['models']['model']['metadata']['deepseek41.attention.layer_norm_rms_epsilon']
(output/'epsilon.txt').write_text(str(epsilon)+'\n')
manifest['rms_epsilon'] = epsilon
manifest['normalization'] = 'attn_q_a_norm.weight after RMS_NORM, matches input to Q_B'
(output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
