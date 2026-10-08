#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
SNAP=ROOT.parent/'stage8/candidate-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text());assert all(sha(SNAP/n)==h for n,h in hashes.items())
order=json.loads((ROOT.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order'].split(',')
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
for k in ['GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE']:env.pop(k,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',LD_LIBRARY_PATH=str(SNAP),GGML_CUDA_DISABLE_GRAPHS='1')
meta={'source_sha256':sha(ROOT/'down-components.cpp'),'binary_sha256':sha(ROOT/'down-components'),'head':subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip(),'snapshot_hashes_verified':len(hashes),'runs':[],'production_change':False}
for name,gpu in [('ada',order[0]),('ampere',order[-1])]:
 env['CUDA_VISIBLE_DEVICES']=gpu
 for dataset in ['down','weighted']:
  directory=ROOT/'inputs'
  mode='post' if dataset.startswith('post') else 'projection'
  label=f'{dataset}-{name}'+('-profile' if args.profile else '')
  output=ROOT/label
  command=[str(ROOT/'down-components'),str(directory),str(output),dataset]
  record={'dataset':dataset,'architecture':name,'gpu':gpu,'command':command,'input_manifest_sha256':sha(directory/'manifest.json')}
  record['weight_index_sha256']=sha(directory/'weight-index.json')
  if args.profile:command=['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--output',str(output),*command]
  record['executed_command']=command
  print('START',label,flush=True)
  with (ROOT/(label+'.log')).open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  record['exit_code']=result.returncode;meta['runs'].append(record)
  (ROOT/('components-profile-manifest.json' if args.profile else 'components-manifest.json')).write_text(json.dumps(meta,indent=2)+'\n')
  if result.returncode:raise SystemExit(result.returncode)
  loaded={line.split()[-1] for line in (output/'loaded-libraries.txt').read_text().splitlines()}
  record['loaded_libraries']={p:sha(p) for p in sorted(loaded)}
  assert loaded and all(Path(p).parent==SNAP and sha(p)==hashes[Path(p).name] for p in loaded)
  (ROOT/('components-profile-manifest.json' if args.profile else 'components-manifest.json')).write_text(json.dumps(meta,indent=2)+'\n')
  print('FINISH',dataset,name,flush=True)
