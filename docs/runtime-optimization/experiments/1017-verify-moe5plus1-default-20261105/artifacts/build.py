from pathlib import Path
import json,subprocess,hashlib
R=Path('/home/sergei/Github/llama.cpp')
P=Path(__file__).resolve().parent
C=P/'candidate';C.mkdir()
old=json.loads((R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts/build.json').read_text())
e=old['compile'][0].copy();cmd=e['command'][:];cmd[cmd.index('-o')+1]=str(C/'deepseek41.cpp.o')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e.update(command=cmd,object=cmd[cmd.index('-o')+1],source_sha256=sha(R/e['source']))
with(P/'compile.log').open('wb')as log:subprocess.run(cmd,cwd=e['cwd'],stdout=log,stderr=subprocess.STDOUT,check=True)
link=old['link'][:];link[link.index('-o')+1]=str(C/'libllama.so.0');link[link.index(old['compile'][0]['object'])]=e['object']
with(P/'link.log').open('wb')as log:subprocess.run(link,cwd=old['link_cwd'],stdout=log,stderr=subprocess.STDOUT,check=True)
(C/'libllama.so').symlink_to('libllama.so.0')
(P/'build.json').write_text(json.dumps(dict(complete=True,compile=[e],link=link,link_cwd=old['link_cwd'],private_library=str(C/'libllama.so.0'),private_llama_sha256=sha(C/'libllama.so.0'),previous_private_sha256=old['private_llama_sha256']),indent=2)+'\n')
print('BUILT',sha(C/'libllama.so.0'))
