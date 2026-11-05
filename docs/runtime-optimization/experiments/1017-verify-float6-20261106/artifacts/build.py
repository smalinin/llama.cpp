from pathlib import Path
import hashlib,json,subprocess,shlex
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent;C=P/'candidate';C.mkdir()
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
commands=json.loads((R/'build-aliceai/compile_commands.json').read_text());e=next(x for x in commands if x['file']==str(R/'src/llama-graph.cpp'));cmd=e.get('arguments')or shlex.split(e['command']);cmd[cmd.index('-o')+1]=str(C/'llama-graph.cpp.o')
with(P/'compile.log').open('wb')as f:subprocess.run(cmd,cwd=e['directory'],stdout=f,stderr=subprocess.STDOUT,check=True)
link=shlex.split((R/'build-aliceai/src/CMakeFiles/llama.dir/link.txt').read_text());link[link.index('-o')+1]=str(C/'libllama.so.0');idx=next(i for i,x in enumerate(link)if x.endswith('/llama-graph.cpp.o'));link[idx]=str(C/'llama-graph.cpp.o')
with(P/'link.log').open('wb')as f:subprocess.run(link,cwd=R/'build-aliceai/src',stdout=f,stderr=subprocess.STDOUT,check=True)
(C/'libllama.so').symlink_to('libllama.so.0')
b=dict(complete=True,command=cmd,cwd=e['directory'],source_sha256=sha(R/'src/llama-graph.cpp'),reference_source_sha256=sha(P/'reference-llama-graph.cpp'),link=link,link_cwd=str(R/'build-aliceai/src'),private_llama_sha256=sha(C/'libllama.so.0'),reference_llama_sha256=sha(R/'build-aliceai/bin/libllama.so'),header_changed=False,cuda_source_changed=False)
(P/'build.json').write_text(json.dumps(b,indent=2)+'\n');(P/'runtime.patch').write_text(subprocess.check_output(['git','diff','--','src'],cwd=R,text=True));print('FLOAT6_BUILT',b['private_llama_sha256'])
