#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53';OLD=R.parent/'stage27/candidate-bin';OUT=R/'candidate-bin'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old=json.loads((R.parent/'stage27/build-manifest.json').read_text())['candidate_hashes']
assert all(sha(OLD/n)==h for n,h in old.items())
assert not OUT.exists();shutil.copytree(OLD,OUT,symlinks=True)
sources=['src/llama-kv-cache-kpool.cpp','src/llama-kv-cache-kpool.h','src/llama-graph.cpp']
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'before_hashes':old,'sources':{n:sha(REPO/n) for n in sources},'commands':[],'reused_objects':{}}
commands=[];compile_db=json.loads((B/'compile_commands.json').read_text())
for name in [n for n in sources if n.endswith('.cpp')]:
 entry=next(e for e in compile_db if e['file']==str(REPO/name));cmd=shlex.split(entry['command']);cmd[cmd.index('-o')+1]=str(R/(Path(name).name+'.o'));commands.append((cmd,entry['directory']))
link=shlex.split((B/'src/CMakeFiles/llama.dir/link.txt').read_text())
for i,a in enumerate(link):
 if a.endswith('.o'):
  p=(B/'src'/a).resolve()
  if p.name in ['llama-kv-cache-kpool.cpp.o','llama-graph.cpp.o']:link[i]=str(R/p.name)
  elif p.name=='llama-kv-cache.cpp.o':
   p=R.parent/'stage27/llama-kv-cache.cpp.o';link[i]=str(p);manifest['reused_objects'][str(p)]=sha(p)
  else:manifest['reused_objects'][str(p)]=sha(p)
 elif a.startswith('../bin/') and '.so' in a:link[i]=str(OLD/Path(a).name)
 elif a.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(OUT)
link[link.index('-o')+1]=str(OUT/'libllama.so.0.4.0');commands.append((link,str(B/'src')))
(R/'source.patch').write_bytes(subprocess.check_output(['git','diff','--',*sources],cwd=REPO))
with (R/'build.log').open('w') as log:
 for cmd,cwd in commands:
  manifest['commands'].append({'command':cmd,'cwd':cwd});p=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
  assert p.returncode==0,p.returncode
manifest.update(exit_code=0,candidate_hashes={n:sha(OUT/n) for n in old})
assert {n for n in old if old[n]!=manifest['candidate_hashes'][n]}=={'libllama.so','libllama.so.0','libllama.so.0.4.0'}
(R/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built libllama; Stage27 tail fix retained; other libraries unchanged.')
