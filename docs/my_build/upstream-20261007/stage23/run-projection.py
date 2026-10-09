#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;S=R.parent/'stage8/candidate-bin';B=R.parent/'stage13/isolated-replay'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
old=json.loads((R.parent/'stage13/isolated-manifest.json').read_text());assert sha(B)==old['binary_sha256']
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text());assert all(sha(S/n)==h for n,h in libs.items())
order=json.loads((R.parent/'stage9/current-server/off/manifest.json').read_text())['gpu_order'].split(',');env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
for k in ['GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','LLAMA_DSV41_DIAGNOSTIC']:env.pop(k,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order[3],LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
cmd=[str(B),str(R/'projection-inputs'),'projection',str(R/'projection-output')]
m={'command':cmd,'binary_sha256':sha(B),'source_sha256':sha(B.with_suffix('.cpp')),'model_device':3,'gpu_uuid':order[3],'input_manifest_sha256':sha(R/'projection-inputs/manifest.json'),'snapshot_sha256':libs,'production_change':False}
with (R/'projection.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
m['exit_code']=p.returncode;(R/'projection-run-manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('projection exit',p.returncode);raise SystemExit(p.returncode)
