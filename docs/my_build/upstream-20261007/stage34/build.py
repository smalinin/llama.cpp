from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,concurrent.futures
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53';BASE=R.parent/'stage33/final-bin';OUT=R/'candidate-bin';OUT.mkdir(exist_ok=True)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if not (OUT/'llama-server').exists():subprocess.run(['cp','-a','--reflink=auto',str(BASE)+'/.',str(OUT)],check=True)
entries=json.loads((B/'compile_commands.json').read_text());manifest={'base_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'sources':{},'commands':[]}
objects={}
for src in ['ggml/src/ggml-cuda/mmf.cu','tools/server/server-context.cpp']:
 entry=next(e for e in entries if e['file']==str(REPO/src));cmd=shlex.split(entry['command']);out=R/(Path(src).name+'.o');cmd[cmd.index('-o')+1]=str(out);objects[Path(src).name+'.o']=out;manifest['sources'][src]=sha(REPO/src);manifest['commands'].append((cmd,entry['directory']))
def run(pair):
 cmd,cwd=pair;name=Path(cmd[cmd.index('-o')+1]).name if '-o' in cmd else 'link';log=R/(name+'.log')
 with log.open('w') as f:res=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 if res.returncode:raise RuntimeError(log.read_text()[-7000:])
 print('done',name,flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(run,manifest['commands']))
archive=R/'libserver-context.a';shutil.copy2(B/'tools/server/libserver-context.a',archive)
subprocess.run(['ar','r',str(archive),str(objects['server-context.cpp.o'])],check=True)
for linkfile,cwd,target in [(B/'ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/link.txt',B/'ggml/src/ggml-cuda','libggml-cuda.so.0.23.0'),(B/'tools/server/CMakeFiles/llama-server-impl.dir/link.txt',B/'tools/server','libllama-server-impl.so')]:
 raw=shlex.split(linkfile.read_text());cmd=[]
 for v in raw:cmd.extend(shlex.split((cwd/v[1:]).read_text()) if v.startswith('@') else [v])
 for i,v in enumerate(cmd):
  if v.endswith('.o'):
   q=(cwd/v).resolve();cmd[i]=str(objects.get(q.name,q))
  elif v.endswith('.a'):
   q=(cwd/v).resolve();cmd[i]=str(archive if q.name=='libserver-context.a' else q)
  elif '.so' in v and not v.startswith('-') and (OUT/Path(v).name).exists():cmd[i]=str(OUT/Path(v).name)
  elif v.startswith('-Wl,-rpath,'):cmd[i]='-Wl,-rpath,'+str(OUT)
 cmd[cmd.index('-o')+1]=str(OUT/target);manifest['commands'].append((cmd,str(cwd)));run((cmd,str(cwd)))
manifest['hashes']={p.name:sha(p) for p in OUT.iterdir() if p.is_file()};(R/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
