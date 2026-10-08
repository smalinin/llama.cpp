import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
S=R.parent/'stage8/candidate-bin'
order=json.loads((R.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order'].split(',')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={'binary_sha256':sha(R/'attention-replay'),'source_sha256':sha(R/'attention-replay.cpp'),'runs':[],'production_change':False}
for label,device in [('ada',order[0]),('ampere',order[-1])]:
 env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=device,LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
 for k in ('NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_FUSION'):env.pop(k,None)
 cmd=[str(R/'attention-replay'),str(R/'layer20-inputs'),str(R/f'layer20-{label}')]
 with (R/f'layer20-{label}.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 manifest['runs'].append({'command':cmd,'gpu_uuid':device,'exit_code':result.returncode})
 assert result.returncode==0,(R/f'layer20-{label}.log').read_text()
(R/'layer20-run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FINISH Ada/Ampere attention',flush=True)
