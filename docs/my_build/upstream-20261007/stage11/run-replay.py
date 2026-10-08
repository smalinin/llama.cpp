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
parser.add_argument('--profile', action='store_true')
parser.add_argument('--no-fusion', action='store_true')
parser.add_argument('--down', action='store_true')
parser.add_argument('--q8', action='store_true')
parser.add_argument('--bench', action='store_true')
args = parser.parse_args()
assert sum([args.down,args.q8,args.bench])<=1
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
label = ('bench-' if args.bench else 'q8-' if args.q8 else 'down-' if args.down else 'ffn-')+args.arch+('-no-fusion' if args.no_fusion else '')+('-profile' if args.profile else '')
if args.no_fusion:
    env['GGML_CUDA_DISABLE_FUSION'] = '1'
binary = 'ffn-bench' if args.bench else 'q8-capture' if args.q8 else 'down-replay' if args.down else 'ffn-replay'
command = [str(ROOT/binary), str(ROOT/'inputs')]
if args.down or args.q8:
    command.append(str(ROOT/('ffn-'+args.arch)))
command.append(str(ROOT/label))
manifest = {'command': command, 'gpu_uuid': uuid, 'snapshot': 'stage8/candidate-bin',
    'binary_sha256': sha(command[0]), 'source_sha256': sha(ROOT/(binary+'.cpp')),
    'input_manifest_sha256': sha(ROOT/'inputs/manifest.json'), 'disable_graphs': True,
    'widths': [1, 2, 4], 'repeats': 2, 'input_columns': 'identical copies', 'inputs_from_capture_widths': [1, 2, 4], 'weights': 'original model tensor contents, all 384 experts', 'fusion_enabled': not args.no_fusion}
if args.down or args.q8:
    manifest['frozen_weights'] = 'inputw1-batch1-repeat routing weights and expert IDs'
    manifest['hidden_sources'] = 'first columns of same-input FFN width1/2/4; identical copies into all down columns'
    manifest['ffn_input_sha256'] = {p.name: sha(p) for p in (ROOT/('ffn-'+args.arch)).glob('inputw1-batch*-repeat-*') if p.name.endswith(('moe-hidden.f32', 'weights.f32', 'ids.i32'))}
manifest['weight_index_sha256'] = sha(ROOT/'inputs/weight-index.json')
if args.bench:
    manifest['benchmark'] = {'warmup':10, 'groups':5,'iterations_per_group':100,'scope':'isolated full FFN, fixed input repeated, CUDA graphs disabled, synchronized host wall time, not model throughput'}
manifest['head'] = subprocess.check_output(['git', '-C', '/home/sergei/Github/llama.cpp', 'rev-parse', 'HEAD'], text=True).strip()
if args.profile:
    command = ['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--output', str(ROOT/label), *command]
manifest['executed_command'] = command
path = ROOT/(label+'-manifest.json')
path.write_text(json.dumps(manifest, indent=2)+'\n')
with (ROOT/(label+'.log')).open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest['exit_code'] = result.returncode
path.write_text(json.dumps(manifest, indent=2)+'\n')
print(args.arch, result.returncode, flush=True)
raise SystemExit(result.returncode)
