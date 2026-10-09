#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,resource,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');LIB=R.parent/'stage28/final-bin'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cmd=['g++','-std=c++17','-O2','-I'+str(REPO/'include'),'-I'+str(REPO/'ggml/include'),str(R/'restore-replay.cpp'),'-L'+str(LIB),'-Wl,-rpath,'+str(LIB),'-lllama','-lggml','-lggml-base','-o',str(R/'restore-replay')]
with (R/'cpu-build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
model=REPO/'build-glm53/tests/test-models/glm5next-moe.gguf';resource.setrlimit(resource.RLIMIT_CORE,(0,0))
manifest={'compile_command':cmd,'model':str(model),'model_sha256':sha(model),'source_sha256':sha(R/'restore-replay.cpp'),'binary_sha256':sha(R/'restore-replay'),'library':str(LIB/'libllama.so.0.4.0'),'library_sha256':sha(LIB/'libllama.so.0.4.0'),'cases':{}}
for label,unified,cache in [('unified-on',1,1),('unified-off',1,0),('separated-on',0,1),('separated-off',0,0)]:
 out=R/'cpu'/label;out.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('LLAMA_','GGML_SCHED_')):env.pop(k)
 env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(LIB),LLAMA_GLM5_POOL_CACHE=str(cache))
 cmd=[str(R/'restore-replay'),str(model),str(unified),str(out)]
 with (out/'rows.jsonl').open('w') as stdout,(out/'run.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=stdout,stderr=log)
 manifest['cases'][label]={'command':cmd,'pool_cache':cache,'kv_unified':bool(unified),'exit_code':p.returncode,'float_sha256':{f.name:sha(f) for f in out.glob('*.f32')}}
 (R/'cpu-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(1)
