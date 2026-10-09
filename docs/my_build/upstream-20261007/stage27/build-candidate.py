#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp');BUILD=REPO/'build-glm53';SNAP=R.parent/'stage8/candidate-bin';OUT=R/'candidate-bin'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/n)==h for n,h in libs.items())
assert sha(BUILD/'bin/libllama.so.0.4.0')==libs['libllama.so.0.4.0']
assert not OUT.exists();shutil.copytree(SNAP,OUT,symlinks=True)
source=REPO/'src/llama-kv-cache.cpp';shutil.copy2(source,R/'llama-kv-cache.cpp')
entry=next(x for x in json.loads((BUILD/'compile_commands.json').read_text()) if x['file']==str(source))
compile_cmd=shlex.split(entry['command']);compile_cmd[compile_cmd.index('-o')+1]=str(R/'llama-kv-cache.cpp.o')
link=shlex.split((BUILD/'src/CMakeFiles/llama.dir/link.txt').read_text());objects={}
for i,arg in enumerate(link):
    if arg.endswith('.o'):
        p=(BUILD/'src'/arg).resolve();objects[str(p)]=sha(p)
        if p.name=='llama-kv-cache.cpp.o':link[i]=str(R/'llama-kv-cache.cpp.o')
    elif arg.startswith('../bin/') and '.so' in arg:link[i]=str(SNAP/Path(arg).name)
    elif arg.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(OUT)
link[link.index('-o')+1]=str(OUT/'libllama.so.0.4.0')
manifest={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
          'source_sha256':sha(source),'snapshot_hashes':libs,'reused_objects':objects,
          'commands':[compile_cmd,link],'command_cwds':[entry['directory'],str(BUILD/'src')]}
(R/'source.patch').write_bytes(subprocess.check_output(['git','-C',str(REPO),'diff','--','src/llama-kv-cache.cpp']))
with (R/'build.log').open('w') as log:
    for command,cwd in zip(manifest['commands'],manifest['command_cwds']):
        p=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise SystemExit(p.returncode)
manifest.update(exit_code=0,candidate_hashes={n:sha(OUT/n) for n in libs})
changed=[n for n,h in libs.items() if manifest['candidate_hashes'][n]!=h]
assert set(changed)=={'libllama.so','libllama.so.0','libllama.so.0.4.0'}
manifest['changed_files']=changed
(R/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Candidate built; only libllama and its aliases differ from snapshot',flush=True)
