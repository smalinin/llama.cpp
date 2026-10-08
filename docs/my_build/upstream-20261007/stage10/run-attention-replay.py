#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent/'stage8/candidate-bin'
parser = argparse.ArgumentParser()
parser.add_argument('arch', choices=['ada', 'ampere'])
args = parser.parse_args()
uuid = {'ada': 'GPU-3972c9ef-dd88-6b95-ac0c-fc9b0eb86f76',
        'ampere': 'GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783'}[args.arch]
env = os.environ.copy()
for key in list(env):
    if key.startswith(('LLAMA_ARG_', 'LLAMA_MTP_', 'LLAMA_DSPARK_', 'GGML_SCHED_')):
        env.pop(key)
for key in ('GGML_CUDA_DISABLE_FUSION', 'NVIDIA_TF32_OVERRIDE', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE'):
    env.pop(key, None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=uuid,
           LD_LIBRARY_PATH=str(SNAPSHOT), GGML_CUDA_DISABLE_GRAPHS='1')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAPSHOT/name) == value for name, value in hashes.items())
command = [str(ROOT/'attention-replay'), str(ROOT), str(ROOT/('attention-'+args.arch))]
manifest = {'command': command, 'gpu_uuid': uuid, 'snapshot': 'stage8/candidate-bin',
    'binary_sha256': sha(command[0]), 'source_sha256': sha(ROOT/'attention-replay.cpp'),
    'input_manifest_sha256': sha(ROOT/'attention-input-manifest.json'), 'disable_graphs': True,
    'widths': [1, 2, 3, 4, 5, 8, 16], 'repeats': 2, 'q_mask_columns': 'identical copies'}
path = ROOT/('attention-'+args.arch+'-manifest.json')
path.write_text(json.dumps(manifest, indent=2)+'\n')
with (ROOT/('attention-'+args.arch+'.log')).open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest['exit_code'] = result.returncode
path.write_text(json.dumps(manifest, indent=2)+'\n')
print(args.arch, result.returncode, flush=True)
raise SystemExit(result.returncode)
