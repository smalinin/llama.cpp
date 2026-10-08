#!/usr/bin/env python3
import os,subprocess,json
from pathlib import Path
R=Path(__file__).resolve().parent
results=[]
def execute(label,command,env):
    print('START',label,flush=True)
    with (R/(label+'.log')).open('w') as output:
        code=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
    record={'case':label,'exit_code':code,'command':command,'env':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION') if k in env}}
    results.append(record);(R/'cuda-test-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('FINISH',label,code,flush=True)
    if code: raise SystemExit(code)
for version in ('before','after'):
    for gpu in ('4','3'):
        env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,LD_LIBRARY_PATH=str(R/(version+'-bin')))
        env.pop('GGML_CUDA_DISABLE_FUSION',None);env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
        execute('check-'+version+'-gpu'+gpu,[str(R/'cuda-check'),'all',str(R/('check-'+version+'-gpu'+gpu+'.bin'))],env)
        execute('norm-ops-'+version+'-gpu'+gpu,[str(R/(version+'-bin/test-backend-ops')),'test','-b','CUDA0','-o','RMS_NORM,SCALE,RMS_NORM_MUL_ADD,RMS_NORM_MUL_ROPE'],env)
    env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4',LD_LIBRARY_PATH=str(R/(version+'-bin')),GGML_CUDA_DISABLE_GRAPHS='1')
    for mode in ('norm','mmvf'):
        label=mode+'-'+version
        execute(label,['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(R/label),str(R/'cuda-check'),mode+'-profile',str(R/(label+'.bin'))],env)
        execute(label+'-stats',['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(R/(label+'.nsys-rep'))],env)
# Compare fused and unfused numerical behavior without graphs.
env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4',LD_LIBRARY_PATH=str(R/'after-bin'),GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_DISABLE_FUSION='1')
execute('check-after-unfused-gpu4',[str(R/'cuda-check'),'correctness',str(R/'check-after-unfused-gpu4.bin')],env)
