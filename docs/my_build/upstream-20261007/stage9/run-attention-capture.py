#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent/'stage8/candidate-bin'
parser=argparse.ArgumentParser()
parser.add_argument('--no-fusion',action='store_true')
args=parser.parse_args()
deadline=time.monotonic()+1800
while True:
    path=ROOT/'n1-server/results.json'
    if path.exists() and len(json.loads(path.read_text()))==2: break
    if time.monotonic()>deadline: raise TimeoutError('n1 server completion')
    time.sleep(1)
model=json.loads((ROOT.parent/'stage7/model-inspection.json').read_text())['models']['model']['path']
order=json.loads((ROOT.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order']
env=os.environ.copy()
for key in list(env):
    if key.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')): env.pop(key)
for key in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE',
            'QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE'):
    env.pop(key,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order,LD_LIBRARY_PATH=str(SNAPSHOT))
if args.no_fusion: env['GGML_CUDA_DISABLE_FUSION']='1'
label='attention-capture'
command=[str(ROOT/'capture-attention'),model,str(ROOT.parent/'stage7/baseline-prompt.txt'),
         str(ROOT/'forced-current-native.i32'),str(ROOT/(label+'-output'))]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAPSHOT/name)==value for name,value in hashes.items())
manifest={'command':command,'gpu_order':order,'snapshot':'stage8/candidate-bin',
          'binary_sha256':sha(command[0]),'source_sha256':sha(ROOT/'capture-attention.cpp'),
          'forced_prefix_sha256':sha(ROOT/'forced-current-native.i32'),'no_fusion':args.no_fusion,
          'cb_eval':'capture layer 2 at first generated token; then FP32 decode-only controls preserving prefill',
          'prefill_chunks':[1139,512,4],'swa_full':False,'output_capacity':4,
          'forced_prefix_source':'current-server/off/baseline-diagnostic-1-response.json'}
(ROOT/(label+'-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print('START',label,'capture plus decode precision controls',flush=True)
with (ROOT/(label+'.log')).open('w') as log:
    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest['exit_code']=result.returncode
(ROOT/(label+'-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print('FINISH',label,result.returncode,flush=True)
raise SystemExit(result.returncode)
