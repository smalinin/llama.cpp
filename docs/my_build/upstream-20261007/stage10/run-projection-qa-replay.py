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
inputs = json.loads((ROOT/'projection-qa-inputs/manifest.json').read_text())
command = [str(ROOT/'projection-qa-replay'), str(ROOT/'projection-qa-inputs'), str(inputs['type']),
           str(inputs['ne'][0]), str(inputs['ne'][1]), str(ROOT/('projection-qa-'+args.arch))]
manifest = {'command': command, 'gpu_uuid': uuid, 'snapshot': 'stage8/candidate-bin',
    'binary_sha256': sha(command[0]), 'source_sha256': sha(ROOT/'projection-qa-replay.cpp'),
    'input_manifest_sha256': sha(ROOT/'projection-qa-inputs/manifest.json'), 'disable_graphs': True,
    'widths': [1, 2, 4], 'repeats': 2, 'input_columns': 'identical copies', 'inputs_from_capture_widths': [1, 2, 4], 'weights': ['original quantized', 'dequantized FP32']}
path = ROOT/('projection-qa-'+args.arch+'-manifest.json')
path.write_text(json.dumps(manifest, indent=2)+'\n')
with (ROOT/('projection-qa-'+args.arch+'.log')).open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest['exit_code'] = result.returncode
path.write_text(json.dumps(manifest, indent=2)+'\n')
print(args.arch, result.returncode, flush=True)
raise SystemExit(result.returncode)
