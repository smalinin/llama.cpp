#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'stage8/candidate-bin'
parser=argparse.ArgumentParser();parser.add_argument('--cases',default='cases.txt');parser.add_argument('--output',default='model-output');parser.add_argument('--prefix',default='replay');args=parser.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text());assert all(sha(S/n)==h for n,h in libs.items())
model=json.loads((R.parent/'stage7/model-inspection.json').read_text())['models']['model']['path']
order=json.loads((R.parent/'stage9/current-server/off/manifest.json').read_text())['gpu_order'];env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
for k in ['GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','LLAMA_DSV41_DIAGNOSTIC']:env.pop(k,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order,LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
cmd=[str(R/'focused-replay'),model,str(R/args.cases),str(R/args.output)]
inputs={str(Path(p)):sha(p) for line in (R/args.cases).read_text().splitlines() for p in line.split()[1:3]}
m={'command':cmd,'gpu_order':order,'environment':{k:env[k] for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','GGML_CUDA_DISABLE_GRAPHS']},'binary_sha256':sha(cmd[0]),'snapshot_sha256':libs,'input_sha256':inputs,'cases_sha256':sha(R/args.cases),'head':subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip(),'production_change':False}
(R/(args.prefix+'-run-manifest.json')).write_text(json.dumps(m,indent=2)+'\n');print('START model replay',flush=True)
with (R/(args.prefix+'.log')).open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
m['exit_code']=p.returncode;(R/(args.prefix+'-run-manifest.json')).write_text(json.dumps(m,indent=2)+'\n');print('FINISH model replay',p.returncode,flush=True);raise SystemExit(p.returncode)
