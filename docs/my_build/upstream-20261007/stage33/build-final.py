from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib,json,os,shlex,shutil,subprocess,time
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53';OLD=R.parent/'stage32/candidate-bin';OUT=R/'final-bin';OBJ=R/'objects';OBJ.mkdir(exist_ok=True)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if not OUT.exists():
 OUT.mkdir();subprocess.run(['cp','-a','--reflink=auto',str(OLD)+'/.',str(OUT)],check=True)
entries=json.loads((B/'compile_commands.json').read_text());commands=[];replaced={};reused={}
link=shlex.split((B/'src/CMakeFiles/llama.dir/link.txt').read_text())
for i,v in enumerate(link):
 if v.endswith('.o'):
  old=(B/'src'/v).resolve();dep=Path(str(old)+'.d');d=dep.read_text() if dep.exists() else ''
  src=next((e for e in entries if str(old) in e.get('output','') or shlex.split(e['command'])[shlex.split(e['command']).index('-o')+1]==v),None)
  if src is None:
   src=next(e for e in entries if (Path(e['directory'])/shlex.split(e['command'])[shlex.split(e['command']).index('-o')+1]).resolve()==old)
  needed=any(n in d for n in ['llama-graph.h','llama-kv-cache.h','llama-kv-cache-dsv4.h']) or src['file'] in [str(REPO/'src/llama-graph.cpp'),str(REPO/'src/llama-kv-cache.cpp'),str(REPO/'src/models/glm5next.cpp')]
  if needed:
   out=OBJ/(str(Path(src['file']).relative_to(REPO/'src')).replace('/','_')+'.o');cmd=shlex.split(src['command']);cmd[cmd.index('-o')+1]=str(out);commands.append({'command':cmd,'cwd':src['directory'],'source':src['file'],'sha256':sha(src['file']),'object':str(out)});link[i]=str(out)
  else:
   if old.name=='llama-kv-cache-kpool.cpp.o':old=R.parent/'stage28/final-llama-kv-cache-kpool.cpp.o'
   link[i]=str(old);reused[str(old)]=sha(old)
 elif v.startswith('../bin/') and '.so' in v:link[i]=str(OUT/Path(v).name)
 elif v.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(OUT)
link[link.index('-o')+1]=str(OUT/'libllama.so.0.4.0')
man={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'commands':commands,'link':link,'reused_objects':reused,'source_changes':{n:sha(REPO/n) for n in subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines() if n.startswith('src/')},'base_sources':{},'before_hashes':{p.name:sha(p) for p in OLD.iterdir() if p.is_file()}}
(R/'build-final-manifest.json').write_text(json.dumps(man,indent=2)+'\n');print('Compiling',len(commands),'objects',flush=True)
def run(c):
 out=Path(c['object']);log=out.with_suffix('.log')
 if out.exists() and out.stat().st_mtime_ns > max(Path(c['source']).stat().st_mtime_ns, (REPO/'src/llama-graph.h').stat().st_mtime_ns, (REPO/'src/llama-kv-cache.h').stat().st_mtime_ns): return Path(c['source']).name
 with log.open('w') as f:result=subprocess.run(c['command'],cwd=c['cwd'],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:print(log.read_text()[-7000:],flush=True);raise RuntimeError(c['source'])
 return Path(c['source']).name
with ThreadPoolExecutor(max_workers=8) as pool:
 for i,future in enumerate(as_completed([pool.submit(run,c) for c in commands])):
  name=future.result()
  if i%10==0:print('Compiled',i+1,name,flush=True)
with (R/'link.log').open('w') as log:subprocess.run(link,cwd=B/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
man.update(candidate_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},exit_code=0)
(R/'build-final-manifest.json').write_text(json.dumps(man,indent=2)+'\n');print('Build complete',flush=True)
