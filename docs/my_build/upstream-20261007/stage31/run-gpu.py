#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,os,resource,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=R/'ram-replay.cpp');p.add_argument('--label',required=True);p.add_argument('--lib',type=Path,required=True);p.add_argument('--unified',type=int,default=1);p.add_argument('--rs',type=int,default=0);p.add_argument('--cache',type=int,default=0);args=p.parse_args();lib=args.lib;binary=R/args.source.stem
if not binary.exists():
 cmd=['g++','-std=c++17','-O2',*['-I'+str(REPO/d) for d in ['include','ggml/include','src']],str(args.source),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-o',str(binary)]
 with (R/(args.source.stem+'-build.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 (R/(args.source.stem+'-build-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
resource.setrlimit(resource.RLIMIT_CORE,(0,0));out=R/'gpu'/args.label;out.mkdir(parents=True,exist_ok=False);model=REPO/'build-glm53/tests/test-models/glm5next-moe.gguf'
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_','GGML_SCHED_')):env.pop(k)
env.update(CUDA_VISIBLE_DEVICES='GPU-3972c9ef-dd88-6b95-ac0c-fc9b0eb86f76',LD_LIBRARY_PATH=str(lib),LLAMA_GLM5_POOL_CACHE=str(args.cache))
cmd=[str(binary),str(model),str(args.unified),str(args.rs),str(out)]
with (out/'trace.jsonl').open('w') as stdout,(out/'run.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=stdout,stderr=log)
m={'command':cmd,'source':str(args.source),'source_sha256':sha(args.source),'model_sha256':sha(model),'library':str(lib/'libllama.so.0.4.0'),'library_sha256':sha(lib/'libllama.so.0.4.0'),'exit_code':result.returncode,'outputs':{f.name:sha(f) for f in out.glob('*.f32')}};(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(args.label,result.returncode,flush=True)
raise SystemExit(result.returncode)
