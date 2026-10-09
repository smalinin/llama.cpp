import subprocess,os,json,hashlib,time
from pathlib import Path
r=Path(__file__).resolve().parent
m=json.loads((r/'before-off/manifest.json').read_text());model=m['command'][2];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=m['env']['CUDA_VISIBLE_DEVICES'],LD_LIBRARY_PATH=str(r/'final-bin'));env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
cases=[('final-gpu-f16',str(r/'models/deepseek41-moe.gguf'),'1','1','0','-','-'),('final-gpu-q8',str(r/'models/deepseek41-moe.gguf'),'1','1','1','-','-'),('final-real',model,'6','1','0',str(r/'long-prompt.bin'),str(r/'long-tokens.bin'))]
results=[]
for label,*args in cases:
 cmd=[str(r/'replay-final'),*args,str(r/(label+'-replay.jsonl'))];start=time.monotonic()
 with (r/(label+'-replay.log')).open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 result={'label':label,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-start};results.append(result);(r/'final-replay-manifest.json').write_text(json.dumps({'cases':results,'harness_source_sha256':hashlib.sha256((r/'replay.cpp').read_bytes()).hexdigest(),'harness_binary_sha256':hashlib.sha256((r/'replay-final').read_bytes()).hexdigest(),'input_sha256':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['long-prompt.bin','long-tokens.bin','models/deepseek41-moe.gguf']},'env':{k:env.get(k) for k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_GRAPHS']},'lib':hashlib.sha256((r/'final-bin/libllama.so.0.4.0').read_bytes()).hexdigest()},indent=2)+'\n');print(result,flush=True)
 assert p.returncode==0,label
 print((r/(label+'-replay.jsonl')).read_text(),flush=True)
