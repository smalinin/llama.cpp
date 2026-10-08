#!/usr/bin/env python3
"""Check TF32 cause and final default compatibility after the experiment."""
import json
import os
from pathlib import Path
import subprocess
import time

R = Path(__file__).resolve().parent
while not (R / 'verification-results.json').exists():
    time.sleep(5)
results = []


def execute(label, command, env):
    print('START', label, flush=True)
    with (R / (label + '.log')).open('w') as log:
        rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    results.append({'case': label, 'command': command, 'exit_code': rc,
                    'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'QWEN4EXP_FUSED_LID',
                                                'LLAMA_FUSED_LID_DISABLE', 'NVIDIA_TF32_OVERRIDE') if k in env}})
    (R / 'final-progress.json').write_text(json.dumps(results, indent=2) + '\n')
    print('FINISH', label, rc, flush=True)
    if rc:
        raise SystemExit(rc)


events = [json.loads(s) for s in (R / 'capture-before-np1/tensors.jsonl').read_text().splitlines()]
index = {(e['step'], e['name']): e for e in events}
env = os.environ.copy()
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='4', LD_LIBRARY_PATH=str(R / 'after-bin'))
for step in (0, 31):
    q, k, score = [index[step, 'indexer_' + label + '-3'] for label in ('q', 'k', 'score_blk')]
    command = [str(R / 'qsa-check'), 'replay',
               *[str(R / 'capture-before-np1' / e['file']) for e in (q, k, score)],
               str(score['ne'][0]), str(score['ne'][1]), str(score['ne'][2])]
    env.pop('NVIDIA_TF32_OVERRIDE', None)
    execute(f'replay-step{step}-tf32', command, env)
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    execute(f'replay-step{step}-fp32', command, env)
env.pop('NVIDIA_TF32_OVERRIDE', None)
order = json.loads((R.parent / 'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order']
env.update(CUDA_VISIBLE_DEVICES=order, LD_LIBRARY_PATH=str(R / 'final-bin'))
env.pop('QWEN4EXP_FUSED_LID', None)
model = '/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
for slots in (1, 2):
    execute(f'capture-final-np{slots}', [str(R / 'capture-qsa'), model, str(R / f'capture-final-np{slots}'),
                                      str(slots), str(R / 'capture-prompt.txt')], env)
# Verify that the final gate also enables the experimental path.
env['QWEN4EXP_FUSED_LID'] = '1'
execute('capture-final-optin-np1', [str(R / 'capture-qsa'), model, str(R / 'capture-final-optin-np1'),
                                  '1', str(R / 'capture-prompt.txt')], env)
env.pop('QWEN4EXP_FUSED_LID')
execute('real-final', ['python3', str(R / 'run-real.py'), 'final'], env)
(R / 'final-results.json').write_text(json.dumps(results, indent=2) + '\n')
