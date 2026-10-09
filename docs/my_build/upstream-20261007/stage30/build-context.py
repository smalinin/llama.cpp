#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,shlex,shutil,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53';OLD=R.parent/'stage28/final-bin'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--source',type=Path,default=REPO/'src/llama-context.cpp');a=p.parse_args();out=R/(a.label+'-bin');obj=R/(a.label+'-llama-context.cpp.o')
prev=json.loads((R.parent/'stage28/final-build-manifest.json').read_text());old=prev['candidate_hashes'];assert all(sha(OLD/n)==h for n,h in old.items());assert not out.exists();shutil.copytree(OLD,out,symlinks=True)
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source':str(a.source),'source_sha256':sha(a.source),'base_sources':prev['sources'],'before_hashes':old,'commands':[],'reused_objects':{}}
entry=next(e for e in json.loads((B/'compile_commands.json').read_text()) if e['file']==str(REPO/'src/llama-context.cpp'));cmd=shlex.split(entry['command']);cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(a.source)
commands=[(cmd,entry['directory'])];link=shlex.split((B/'src/CMakeFiles/llama.dir/link.txt').read_text())
for i,v in enumerate(link):
 if v.endswith('.o'):
  q=(B/'src'/v).resolve()
  if q.name=='llama-context.cpp.o':link[i]=str(obj);continue
  if q.name=='llama-kv-cache.cpp.o':q=R.parent/'stage27/llama-kv-cache.cpp.o'
  if q.name in ['llama-graph.cpp.o','llama-kv-cache-kpool.cpp.o']:q=R.parent/'stage28'/('final-'+q.name)
  link[i]=str(q);manifest['reused_objects'][str(q)]=sha(q)
 elif v.startswith('../bin/') and '.so' in v:link[i]=str(OLD/Path(v).name)
 elif v.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(out)
link[link.index('-o')+1]=str(out/'libllama.so.0.4.0');commands.append((link,str(B/'src')))
with (R/(a.label+'-build.log')).open('w') as log:
 for command,cwd in commands:
  manifest['commands'].append({'command':command,'cwd':cwd});r=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT);assert r.returncode==0
manifest.update(exit_code=0,candidate_hashes={n:sha(out/n) for n in old});(R/(a.label+'-build-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n');print(a.label,'built',flush=True)
