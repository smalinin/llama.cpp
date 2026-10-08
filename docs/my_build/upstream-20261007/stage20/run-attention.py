from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent;S=R.parent/'stage8/candidate-bin';B=R.parent/'stage18/attention-replay'
order=json.loads((R.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order'].split(',')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
profile='--profile' in sys.argv
manifest={'binary':str(B),'binary_sha256':sha(B),'source':str(B.with_suffix('.cpp')),'source_sha256':sha(B.with_suffix('.cpp')),'runs':[],'production_change':False}
for name,index in [('ada0',0),('ada2',2),('ampere',5)]:
 env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order[index],LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
 for k in ('NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_FUSION'):env.pop(k,None)
 label=('attention-profile-' if profile else 'attention-')+name
 cmd=[str(B),str(R/'attention-inputs'),str(R/label)]
 if profile:cmd=['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(R/f'profile-{name}')]+cmd
 with (R/f'{label}.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 item={'name':name,'model_device':index,'gpu_uuid':order[index],'command':cmd,'exit_code':p.returncode}
 assert p.returncode==0,(R/f'{label}.log').read_text(errors='replace')
 if profile:
  export=[cmd[0],'export','--type=sqlite','--force-overwrite=true','-o',str(R/f'profile-{name}.sqlite'),str(R/f'profile-{name}.nsys-rep')]
  with (R/f'profile-{name}-export.log').open('w') as log:q=subprocess.run(export,stdout=log,stderr=subprocess.STDOUT)
  assert q.returncode==0
  item.update(export_command=export,export_exit_code=q.returncode)
 manifest['runs'].append(item)
(R/('profile-run-manifest.json' if profile else 'attention-run-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print('FINISH', 'profile' if profile else 'attention',flush=True)
