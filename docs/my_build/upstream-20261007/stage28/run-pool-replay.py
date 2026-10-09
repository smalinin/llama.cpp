#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,resource,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');OLD=R.parent/'stage27/candidate-bin';NEW=R/'candidate-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
command=['g++','-std=c++17','-O2','-I'+str(REPO/'include'),'-I'+str(REPO/'ggml/include'),str(R/'pool-replay.cpp'),'-L'+str(NEW),'-Wl,-rpath,'+str(NEW),'-lllama','-lggml','-lggml-base','-o',str(R/'pool-replay')]
with (R/'pool-build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
resource.setrlimit(resource.RLIMIT_CORE,(0,0));model=REPO/'build-glm53/tests/test-models/glm5next-moe.gguf'
manifest={'command':command,'source_sha256':sha(R/'pool-replay.cpp'),'model':str(model),'model_sha256':sha(model),'cases':{}}
for label,lib,unified,cache in [('before-unified',OLD,1,1),('after-unified',NEW,1,1),('reference-unified',NEW,1,0),('before-separated',OLD,0,1),('after-separated',NEW,0,1)]:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(lib),LLAMA_GLM5_POOL_CACHE=str(cache))
 cmd=[str(R/'pool-replay'),str(model),str(unified),str(R/(label+'.f32'))]
 with (R/(label+'.jsonl')).open('w') as out,(R/(label+'.log')).open('w') as log:p=subprocess.run(cmd,env=env,stdout=out,stderr=log)
 item={'command':cmd,'library_directory':str(lib),'libllama_sha256':sha(lib/'libllama.so.0.4.0'),'pool_cache':cache,'exit_code':p.returncode,'output_sha256':sha(R/(label+'.f32'))};manifest['cases'][label]=item
 (R/'pool-run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(label,p.returncode,flush=True)
