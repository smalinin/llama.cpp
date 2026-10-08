import json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'stage8/candidate-bin'
order=json.loads((R.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order'].split(',')
nsys='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'
manifest=[]
for arch,uuid in [('ada',order[0]),('ampere',order[-1])]:
 env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=uuid,LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
 for k in ('NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_FUSION'):env.pop(k,None)
 cmd=[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(R/f'profile-{arch}'),str(R/'attention-replay'),str(R/'attention-inputs'),str(R/f'attention-profile-{arch}')]
 with (R/f'profile-{arch}.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,(R/f'profile-{arch}.log').read_text()
 export=[nsys,'export','--type=sqlite','--force-overwrite=true','-o',str(R/f'profile-{arch}.sqlite'),str(R/f'profile-{arch}.nsys-rep')]
 with (R/f'profile-{arch}-export.log').open('w') as log:q=subprocess.run(export,stdout=log,stderr=subprocess.STDOUT)
 assert q.returncode==0
 manifest.append({'arch':arch,'gpu_uuid':uuid,'command':cmd,'exit_code':p.returncode,'export_command':export,'export_exit_code':q.returncode})
(R/'profile-run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FINISH profiles',flush=True)
