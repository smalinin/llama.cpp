#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');BIN=REPO/'build-glm53/bin';LIB=R/'final-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(LIB));env.pop('GGML_SCHED_DEBUG_REALLOC',None);env.pop('LLAMA_GLM5_POOL_CACHE',None)
cases={'arch-glm5next':[str(BIN/'test-llama-archs'),'-a','glm5next','-s','1234'],
       'rollback-glm5next':[str(BIN/'test-recurrent-state-rollback'),'-m',str(REPO/'build-glm53/tests/test-models/glm5next-moe.gguf'),'-ngl','0']}
result={}
for name,cmd in cases.items():
 with (R/('final-'+name+'.log')).open('w') as log:p=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 result[name]={'command':cmd,'binary_sha256':sha(cmd[0]),'exit_code':p.returncode,'libllama_sha256':sha(LIB/'libllama.so.0.4.0'),'cuda_visible_devices':''}
 (R/'final-existing-tests.json').write_text(json.dumps(result,indent=2)+'\n');print(name,p.returncode,flush=True)
