#!/usr/bin/env python3
"""Serialize GPU checks after the baseline finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
results = []
if '--resume' in sys.argv:
    history = R / 'verification-progress.json'
    if not history.exists():
        history = R / 'excluded-mask-exact-check/verification-progress.json'
    previous = json.loads(history.read_text())
    results = [r for r in previous if r['exit_code'] == 0]


def execute(label, command, env):
    if any(r['case'] == label for r in results):
        print('ALREADY PASSED', label, flush=True)
        return
    print('START', label, flush=True)
    with (R / (label + '.log')).open('w') as log:
        rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    result = {'case': label, 'exit_code': rc, 'command': command,
              'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'LLAMA_FUSED_LID_DISABLE') if k in env}}
    results.append(result)
    (R / 'verification-progress.json').write_text(json.dumps(results, indent=2) + '\n')
    print('FINISH', label, rc, flush=True)
    if rc:
        raise SystemExit(rc)


while True:
    p = R / 'real-before-summary.json'
    if p.exists():
        data = json.loads(p.read_text())
        if len(data) == 8:
            assert all(x['status'] == 'passed' for x in data)
            break
    if 'failed' in (R / 'real-before.log').read_text():
        raise RuntimeError('baseline failed')
    time.sleep(5)

for gpu in ('4', '3'):
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu, LD_LIBRARY_PATH=str(R / 'after-bin'))
    execute('backend-indexer-gpu' + gpu,
            [str(R / 'after-bin/test-backend-ops'), 'test', '-b', 'CUDA0', '-o', 'LIGHTNING_INDEXER,KPOOL_EXPAND'], env)
    execute('qsa-numerical-gpu' + gpu, [str(R / 'qsa-check'), 'check'], env)
env.update(CUDA_VISIBLE_DEVICES='4')
execute('qsa-memory-reserve', [str(R / 'qsa-check'), 'reserve'], env)

order = json.loads((R.parent / 'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order']
env.update(CUDA_VISIBLE_DEVICES=order)
model = '/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
prompt = R / 'capture-prompt.txt'
prompt.write_text('Read the numbered facts.\n' + ''.join(
    f'Entry {i}: the checkpoint value is {1000+i}; the label is blue.\n' for i in range(200)
) + '\nExplain the checks and list the first ten values.\nAnswer:\n')
for slots in (1, 2):
    for version in ('before', 'after'):
        env.update(LD_LIBRARY_PATH=str(R / (version + '-bin')))
        execute(f'capture-{version}-np{slots}',
                [str(R / 'capture-qsa'), model, str(R / f'capture-{version}-np{slots}'), str(slots), str(prompt)], env)
execute('real-after', ['python3', str(R / 'run-real.py'), 'after'], os.environ.copy())
(R / 'verification-results.json').write_text(json.dumps(results, indent=2) + '\n')
