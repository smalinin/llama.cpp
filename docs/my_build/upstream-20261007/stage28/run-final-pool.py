#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');LIB=R/'final-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={}
for label,unified,cache in [('final-pool-unified',1,1),('final-pool-reference',1,0),('final-pool-separated',0,1)]:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(LIB),LLAMA_GLM5_POOL_CACHE=str(cache))
 cmd=[str(R/'pool-replay'),str(REPO/'build-glm53/tests/test-models/glm5next-moe.gguf'),str(unified),str(R/(label+'.f32'))]
 with (R/(label+'.jsonl')).open('w') as out,(R/(label+'.log')).open('w') as log:p=subprocess.run(cmd,env=env,stdout=out,stderr=log)
 result[label]={'command':cmd,'pool_cache':cache,'exit_code':p.returncode,'libllama_sha256':sha(LIB/'libllama.so.0.4.0'),'output_sha256':sha(R/(label+'.f32'))};assert p.returncode==0
 (R/'final-pool-run-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(label,p.returncode,flush=True)
