#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', ROOT.parent / 'stage7/run-dspark.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original = Path(runner.base.PROFILES['deepseek41']['draft'])
q4k = ROOT / 'DeepSeek-V4.1-Flash-DSpark-Q4_K-from-MXFP4.gguf'
inspection_spec = importlib.util.spec_from_file_location('inspection', ROOT.parent / 'stage7/inspect-models.py')
inspection = importlib.util.module_from_spec(inspection_spec)
inspection_spec.loader.exec_module(inspection)
models = {name: inspection.inspect(path) for name, path in [('mxfp4', original), ('q4k', q4k)]}
for key in models['mxfp4']['metadata']:
    if key not in ('general.file_type', 'general.quantization_version'):
        assert models['mxfp4']['metadata'][key] == models['q4k']['metadata'][key], key
left = {t['name']: t for t in models['mxfp4']['tensors']}
right = {t['name']: t for t in models['q4k']['tensors']}
assert left.keys() == right.keys()
changed = []
for name in left:
    a, b = left[name], right[name]
    assert a['shape'] + [1]*(4-len(a['shape'])) == b['shape'] + [1]*(4-len(b['shape'])), name
    if a['ggml_type'] == 39:
        assert b['ggml_type'] == 12, name
        changed.append(name)
    else:
        assert a['ggml_type'] == b['ggml_type'], name
assert len(changed) == 9
manifest = {'models': models, 'changed_tensors': changed, 'requantization': True,
            'q4k_sha256': hashlib.file_digest(q4k.open('rb'), 'sha256').hexdigest(),
            'original_sha256': hashlib.file_digest(original.open('rb'), 'sha256').hexdigest(),
            'order': ['q4k', 'mxfp4'], 'config': 'n3-p0'}
(ROOT / 'q4-comparison-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
results = []
for label, path in [('q4k', q4k), ('mxfp4', original)]:
    runner.base.PROFILES['deepseek41']['draft'] = path
    args = argparse.Namespace(output=ROOT / 'q4-comparison' / label, verbosity=3,
                              profile=False, tf32_off=False,
                              diagnostic_repeats=2, diagnostic_only=False)
    result = runner.run('n3-p0', args)
    result['draft_variant'] = label
    results.append(result)
    (ROOT / 'q4-comparison/results.json').write_text(json.dumps(results, indent=2) + '\n')
