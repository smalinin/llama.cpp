#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,resource,shlex,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53';OLD=R.parent/'stage27/candidate-bin';FINAL=R/'final-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=next(e for e in json.loads((B/'compile_commands.json').read_text()) if e['file']==str(REPO/'tests/test-llama-archs.cpp'))
cmd=shlex.split(e['command']);cmd[cmd.index('-o')+1]=str(R/'mtp-replay.o');cmd[cmd.index(e['file'])]=str(R/'mtp-replay.cpp')
link=shlex.split((B/'tests/CMakeFiles/test-llama-archs.dir/link.txt').read_text())
for i,a in enumerate(link):
 if a.endswith('.o'):link[i]=str(R/'mtp-replay.o')
 elif a.startswith('../bin/') and '.so' in a:link[i]=str(FINAL/Path(a).name)
 elif a.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(FINAL)
link[link.index('-o')+1]=str(R/'mtp-replay')
with (R/'mtp-build.log').open('w') as log:
 for c,cwd in [(cmd,e['directory']),(link,str(B/'tests'))]:subprocess.run(c,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
resource.setrlimit(resource.RLIMIT_CORE,(0,0));result={'commands':[cmd,link],'source_sha256':sha(R/'mtp-replay.cpp'),'fixture_source_sha256':sha(REPO/'tests/test-llama-archs.cpp'),'cases':{}}
cases=[('mtp-before-unified',OLD,1,3,1),('mtp-final-unified',FINAL,1,3,1),('mtp-reference-unified',FINAL,1,3,0),('mtp-before-separated',OLD,0,3,1),('mtp-final-separated',FINAL,0,3,1),('mtp-before-single',OLD,1,1,1),('mtp-final-single',FINAL,1,1,1)]
for name,lib,unified,nseq,share in cases:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(lib),LLAMA_GLM5_MTP_TOPK_SHARE=str(share));env.pop('LLAMA_GLM5_POOL_CACHE',None)
 command=[str(R/'mtp-replay'),str(unified),str(nseq),str(R/(name+'.f32'))]
 with (R/(name+'.jsonl')).open('w') as out,(R/(name+'.log')).open('w') as log:p=subprocess.run(command,env=env,stdout=out,stderr=log)
 result['cases'][name]={'command':command,'library_directory':str(lib),'libllama_sha256':sha(lib/'libllama.so.0.4.0'),'topk_share':share,'exit_code':p.returncode,'output_sha256':sha(R/(name+'.f32'))}
 (R/'mtp-run-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(name,p.returncode,flush=True)
