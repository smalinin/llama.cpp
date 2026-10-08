#!/usr/bin/env python3
"""Verify the final pooling order and explicit opt-in on the review binary."""
import json
import os
from pathlib import Path
import subprocess
import time

R = Path(__file__).resolve().parent
while not (R / 'final-results.json').exists():
    time.sleep(5)
results = []


def execute(label, command, env):
    print('START', label, flush=True)
    with (R / (label + '.log')).open('w') as log:
        rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    results.append({'case': label, 'command': command, 'exit_code': rc,
                    'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'QWEN4EXP_FUSED_LID') if k in env}})
    (R / 'review-progress.json').write_text(json.dumps(results, indent=2) + '\n')
    print('FINISH', label, rc, flush=True)
    if rc:
        raise SystemExit(rc)


env = os.environ.copy()
order = json.loads((R.parent / 'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order']
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=order, LD_LIBRARY_PATH=str(R / 'review-bin'))
model = '/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
env.pop('QWEN4EXP_FUSED_LID', None)
execute('capture-review-default-np1', [str(R / 'capture-qsa'), model, str(R / 'capture-review-default-np1'),
                                    '1', str(R / 'capture-prompt.txt')], env)
env['QWEN4EXP_FUSED_LID'] = '1'
for slots in (1, 2):
    execute(f'capture-review-np{slots}', [str(R / 'capture-qsa'), model, str(R / f'capture-review-np{slots}'),
                                      str(slots), str(R / 'capture-prompt.txt')], env)
execute('real-review', ['python3', str(R / 'run-real.py'), 'review', '--contexts', '65536'], env)
(R / 'review-results.json').write_text(json.dumps(results, indent=2) + '\n')
