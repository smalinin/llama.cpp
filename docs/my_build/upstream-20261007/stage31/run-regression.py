from pathlib import Path
import hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53';out=R/'regression';out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(R/'candidate-bin'),CUDA_VISIBLE_DEVICES='',LLAMA_GLM5_POOL_CACHE='1',OMP_NUM_THREADS='2')
cases=[('batch',[str(B/'bin/test-batch-alloc')]),('sparse',[str(B/'bin/test-glm5next-sparse')])]
for kind in ['f16','q8_0']:
 cases.append(('state-'+kind,[str(B/'bin/test-save-load-state'),'-m',str(B/'tests/test-models/glm5next-moe.gguf'),'-ngl','0','-t','2','-tb','2','-n','4','-ctk',kind,'-ctv',kind]))
cases.append(('rollback',[str(B/'bin/test-recurrent-state-rollback'),'-m',str(B/'tests/test-models/glm5next-moe.gguf'),'-ngl','0','-t','2','-tb','2']))
results=[]
for label,cmd in cases:
 with (out/(label+'.log')).open('w') as log:p=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
 results.append({'label':label,'command':cmd,'exit_code':p.returncode});(R/'regression-results.json').write_text(json.dumps(results,indent=2)+'\n');print(label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
