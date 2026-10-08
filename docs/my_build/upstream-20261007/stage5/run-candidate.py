#!/usr/bin/env python3
"""Validate the final derived mask and its model memory effect."""
import json
import os
from pathlib import Path
import subprocess

R = Path(__file__).resolve().parent
assert (R / 'review-results.json').exists()
results = []


def execute(label, command, env):
    print('START', label, flush=True)
    with (R / (label + '.log')).open('w') as log:
        rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    results.append({'case': label, 'command': command, 'exit_code': rc,
                    'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'QWEN4EXP_FUSED_LID') if k in env}})
    (R / 'candidate-progress.json').write_text(json.dumps(results, indent=2) + '\n')
    print('FINISH', label, rc, flush=True)
    if rc:
        raise SystemExit(rc)


env = os.environ.copy()
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', LD_LIBRARY_PATH=str(R / 'candidate-bin'))
for gpu in ('4', '3'):
    env['CUDA_VISIBLE_DEVICES'] = gpu
    execute('qsa-derived-mask-gpu' + gpu, [str(R / 'qsa-check'), 'check'], env)
env['CUDA_VISIBLE_DEVICES'] = '4'
execute('qsa-derived-memory-reserve', [str(R / 'qsa-check'), 'reserve'], env)
env['CUDA_VISIBLE_DEVICES'] = json.loads((R.parent / 'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order']
model = '/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
env.pop('QWEN4EXP_FUSED_LID', None)
execute('capture-candidate-default-np1', [str(R / 'capture-qsa'), model, str(R / 'capture-candidate-default-np1'),
                                       '1', str(R / 'capture-prompt.txt')], env)
env['QWEN4EXP_FUSED_LID'] = '1'
for slots in (1, 2):
    execute(f'capture-candidate-np{slots}', [str(R / 'capture-qsa'), model, str(R / f'capture-candidate-np{slots}'),
                                         str(slots), str(R / 'capture-prompt.txt')], env)
execute('real-candidate', ['python3', str(R / 'run-real.py'), 'candidate', '--contexts', '65536'], env)
(R / 'candidate-results.json').write_text(json.dumps(results, indent=2) + '\n')
