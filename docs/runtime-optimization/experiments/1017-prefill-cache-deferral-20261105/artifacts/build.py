from pathlib import Path
import json,hashlib,subprocess,shlex,concurrent.futures,shutil
R=Path('/home/sergei/Github/llama.cpp');B=R/'build-aliceai';P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';C=P/'candidate';C.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();commands=json.loads((B/'compile_commands.json').read_text());build=[];replacements={}
for rel in ['src/models/deepseek41.cpp','src/llama-kv-cache-dsv4.cpp']:
 x=next(x for x in commands if Path(x['file']).resolve()==R/rel);cmd=x.get('arguments')or shlex.split(x['command']);old=Path(x['directory'])/cmd[cmd.index('-o')+1];out=C/(Path(rel).name+'.o');cmd[cmd.index('-o')+1]=str(out);build.append({'source':rel,'source_sha256':sha(R/rel),'command':cmd,'cwd':x['directory'],'object':str(out)});replacements[old.resolve()]=out
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 def run(x):
  with(P/(Path(x['source']).name+'.build.log')).open('wb')as f:subprocess.run(x['command'],cwd=x['cwd'],stdout=f,stderr=subprocess.STDOUT,check=True)
 list(pool.map(run,build))
cmd=shlex.split((B/'src/CMakeFiles/llama.dir/link.txt').read_text());cmd[cmd.index('-o')+1]=str(C/'libllama.so.0');seen=0
for i,arg in enumerate(cmd):
 p=(B/'src'/arg).resolve()
 if p in replacements:cmd[i]=str(replacements[p]);seen+=1
assert seen==2
with(P/'link.log').open('wb')as f:subprocess.run(cmd,cwd=B/'src',stdout=f,stderr=subprocess.STDOUT,check=True)
(C/'libllama.so').symlink_to('libllama.so.0');(C/'libllama.so.0.4.0').symlink_to('libllama.so.0')
(P/'build.json').write_text(json.dumps({'compile':build,'link':cmd,'link_cwd':str(B/'src'),'header_sha256':sha(R/'src/llama-kv-cache-dsv4.h'),'private_llama_sha256':sha(C/'libllama.so.0'),'production_llama_sha256':sha(R/'build-aliceai/bin/libllama.so.0'),'production_base_sha256':sha(R/'build-aliceai/bin/libggml-base.so.0'),'note':'Only2changed CXX objects in a private llama library. Existing other build objects and production GGML modules unchanged; class layout unchanged, only a forwarding type accessor added. No installed library replacement/default change.'},indent=2)+'\n')
(P/'runtime.patch').write_text(subprocess.check_output(['git','diff','--','src/models/deepseek41.cpp','src/llama-kv-cache-dsv4.cpp','src/llama-kv-cache-dsv4.h'],cwd=R,text=True));shutil.copy2('/tmp/build-prefill-cache-deferral.py',P/'build.py')
print('Private prefill deferral library built.',flush=True)
