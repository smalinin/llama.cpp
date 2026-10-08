#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
finished = ROOT / 'q4-comparison/results.json'
while not finished.exists() or len(json.loads(finished.read_text())) != 2:
    time.sleep(2)
old = ROOT.parent / 'stage5/candidate-bin'
new = ROOT / 'candidate-bin'
build = Path('/home/sergei/Github/llama.cpp/build-glm53/bin')
shutil.copy2(build / 'test-backend-ops', new / 'test-backend-ops')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in new.iterdir() if p.is_file()}
(ROOT / 'candidate-binary-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
source = json.loads((ROOT / 'candidate-source.json').read_text())
repo = Path('/home/sergei/Github/llama.cpp')
source['source_sha256'] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                           for name in source['source_sha256']}
(ROOT / 'candidate-source.json').write_text(json.dumps(source, indent=2) + '\n')
for name in ('precision-tests-before.log', 'precision-tests-after.log', 'precision-tests.json'):
    (ROOT / name).rename(ROOT / ('initial-cpu-reference-' + name))
before = Path('/tmp/llama-stage8-before-tests')
shutil.copy2(new / 'test-backend-ops', before / 'test-backend-ops')
results = []
for label, bin_dir, lib_dir in [('before', before, old), ('after', new, new)]:
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES='GPU-3972c9ef-dd88-6b95-ac0c-fc9b0eb86f76',
               LD_LIBRARY_PATH=str(lib_dir))
    for key in ('GGML_CUDA_DISABLE_FUSION', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'NVIDIA_TF32_OVERRIDE'):
        env.pop(key, None)
    cmd = [str(bin_dir / 'test-backend-ops'), 'test', '-b', 'CUDA0', '-o', 'MUL_MAT', '-p', 'prec=f32']
    with (ROOT / f'precision-tests-{label}.log').open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append({'label': label, 'command': cmd, 'ld_library_path': str(lib_dir),
                    'exit_code': result.returncode})
    assert result.returncode == (1 if label == 'before' else 0), results[-1]
(ROOT / 'precision-tests.json').write_text(json.dumps(results, indent=2) + '\n')
print('FP32 regression controls passed', flush=True)
subprocess.run(['python3', str(ROOT / 'run-precision-replay.py')], check=True)
print('Full-model precision replay completed', flush=True)
