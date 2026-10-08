from pathlib import Path
import hashlib,json,os,subprocess
R=Path(__file__).resolve().parent
old=R.parent/'stage17';S=R.parent/'stage8/candidate-bin'
model=json.loads((R.parent/'stage7/model-inspection.json').read_text())['models']['model']['path']
order=json.loads((R.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order']
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
for k in ('GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE'):env.pop(k,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order,LD_LIBRARY_PATH=str(S),GGML_CUDA_DISABLE_GRAPHS='1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text());assert all(sha(S/n)==h for n,h in libs.items())
configs=[('candidate',old/'free-runs/snapshot-off/explain-prompt.txt',old/'explain-native.i32')]
if os.environ.get('STAGE18_BASELINE')=='1':configs=[('baseline',R.parent/'stage7/baseline-prompt.txt',R.parent/'stage9/forced-current-native.i32')]
for label,prompt,prefix in configs:
 cmd=[str(R/f'{label}-replay'),model,str(prompt),str(prefix),str(R/f'{label}-output')]
 m={'command':cmd,'gpu_order':order,'binary_sha256':sha(cmd[0]),'source_sha256':sha(R/f'{label}-replay.cpp'),'callback_sha256':sha(R/'candidate-callback.h'),'prompt_sha256':sha(prompt),'prefix_sha256':sha(prefix),'snapshot_hashes_verified':len(libs),'production_change':False,'scope':'Only first raw-cache padding boundary at absolute position256'}
 p=R/f'{label}-run-manifest.json';p.write_text(json.dumps(m,indent=2)+'\n')
 print('START',label,flush=True)
 with (R/f'{label}.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 m['exit_code']=result.returncode;p.write_text(json.dumps(m,indent=2)+'\n')
 print('FINISH',label,result.returncode,flush=True);assert result.returncode==0
