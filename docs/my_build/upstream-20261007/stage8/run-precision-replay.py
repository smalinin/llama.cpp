#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'stage7'
SNAPSHOT = ROOT / 'candidate-bin'
model = json.loads((PREVIOUS / 'model-inspection.json').read_text())['models']['model']['path']
order = json.loads((PREVIOUS / 'runs/off/manifest.json').read_text())['gpu_order']
env = os.environ.copy()
for key in list(env):
    if key.startswith(('LLAMA_ARG_', 'LLAMA_MTP_', 'LLAMA_DSPARK_', 'GGML_SCHED_')):
        env.pop(key)
for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION',
            'LLAMA_FUSED_LID_DISABLE', 'QWEN4EXP_FUSED_LID', 'NVIDIA_TF32_OVERRIDE',
            'GGML_CUDA_CUBLAS_COMPUTE_TYPE'):
    env.pop(key, None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=order,
           LD_LIBRARY_PATH=str(SNAPSHOT))
command = [str(PREVIOUS / 'target-replay-server-matched'), model,
           str(PREVIOUS / 'baseline-prompt.txt'), str(PREVIOUS / 'forced-native.i32'),
           str(ROOT / 'precision-replay-output')]
manifest = {'command': command, 'gpu_order': order, 'ld_library_path': str(SNAPSHOT),
            'binary_sha256': hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256((PREVIOUS / 'target-replay.cpp').read_bytes()).hexdigest(),
            'swa_full': False, 'no_perf': False, 'prefill_chunks': [1139, 512, 4],
            'forced_tokens_source': 'Stage7 native; identical across all widths',
            'includes_pending_continue_fix': True}
(ROOT / 'precision-replay-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with (ROOT / 'precision-replay.log').open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest['exit_code'] = result.returncode
(ROOT / 'precision-replay-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
raise SystemExit(result.returncode)
