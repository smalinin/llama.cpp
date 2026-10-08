import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
S=R.parent/'stage8/candidate-bin'
old=R.parent/'stage17'
model=json.loads((R.parent/'stage7/model-inspection.json').read_text())['models']['model']['path']
order=json.loads((R.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order']
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
for k in ('GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE'):env.pop(k,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order,LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(S/name)==value for name,value in libs.items())
cmd=[str(R/'capture-replay'),model,str(old/'free-runs/snapshot-off/explain-prompt.txt'),str(old/'explain-native.i32'),str(R/'capture-output')]
m={'command':cmd,'gpu_order':order,'environment':{k:env[k] for k in ('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','GGML_CUDA_DISABLE_GRAPHS')},'binary_sha256':sha(cmd[0]),'prompt_sha256':sha(cmd[2]),'prefix_sha256':sha(cmd[3]),'snapshot_sha256':libs,'production_change':False}
p=R/'capture-run-manifest.json';p.write_text(json.dumps(m,indent=2)+'\n')
print('START boundary capture',flush=True)
with (R/'capture.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
m['exit_code']=result.returncode;p.write_text(json.dumps(m,indent=2)+'\n')
print('FINISH',result.returncode,flush=True)
raise SystemExit(result.returncode)
