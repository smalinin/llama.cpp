#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent/'stage8/candidate-bin'
model = json.loads((ROOT.parent/'stage7/model-inspection.json').read_text())['models']['model']['path']
order = json.loads((ROOT.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order']
env = os.environ.copy()
for key in list(env):
    if key.startswith(('LLAMA_ARG_', 'LLAMA_MTP_', 'LLAMA_DSPARK_', 'GGML_SCHED_')):
        env.pop(key)
for key in ('GGML_CUDA_DISABLE_FUSION', 'LLAMA_FUSED_LID_DISABLE', 'QWEN4EXP_FUSED_LID',
            'NVIDIA_TF32_OVERRIDE', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE'):
    env.pop(key, None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=order,
           LD_LIBRARY_PATH=str(SNAPSHOT), GGML_CUDA_DISABLE_GRAPHS='1')
command = [str(ROOT/'target-matmul-controls'), model,
           str(ROOT.parent/'stage7/baseline-prompt.txt'),
           str(ROOT.parent/'stage9/forced-current-native.i32'), str(ROOT/'target-matmul-output')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAPSHOT/name) == value for name, value in hashes.items())
manifest = {'command': command, 'gpu_order': order, 'snapshot': 'stage8/candidate-bin',
    'head': subprocess.check_output(['git', '-C', '/home/sergei/Github/llama.cpp', 'rev-parse', 'HEAD'], text=True).strip(),
    'binary_sha256': sha(command[0]), 'source_sha256': sha(ROOT/'target-matmul-controls.cpp'),
    'forced_prefix_sha256': sha(command[3]), 'prompt_sha256': sha(command[2]),
    'disable_graphs': True, 'prefill_chunks': [1139, 512, 4], 'swa_full': False, 'output_capacity': 4,
    'variants': ['decode-scalar-fa-upgate-router','decode-scalar-fa-upgate-hc-router','decode-scalar-fa-upgate-float2d'], 'widths': [1, 2, 4],
    'control': 'all scalar modes must match native on all65 logit rows',
    'experiment': 'after prefill, add scalar router, HC+router, or all ordinary2d floating matmuls to the FP32/scalar FA/upgate control; no production flags',
    'production_change': False}
path = ROOT/'target-matmul-manifest.json'
path.write_text(json.dumps(manifest, indent=2)+'\n')
print('START target-matmul', flush=True)
with (ROOT/'target-matmul.log').open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest['exit_code'] = result.returncode
path.write_text(json.dumps(manifest, indent=2)+'\n')
print('FINISH target-matmul', result.returncode, flush=True)
raise SystemExit(result.returncode)
