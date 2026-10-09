#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
cmd=['g++','-std=c++17','-O2',f'-I{REPO}/include',f'-I{REPO}/src',f'-I{REPO}/ggml/include',str(R/'tail-rollback.cpp'),f'-L{SNAP}',f'-Wl,-rpath,{SNAP}','-lllama','-lggml','-lggml-base','-o',str(R/'tail-rollback')]
p=subprocess.run(cmd,capture_output=True,text=True);(R/'tail-build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
result={'command':cmd,'build_exit_code':p.returncode,'source_sha256':hashlib.sha256((R/'tail-rollback.cpp').read_bytes()).hexdigest(),'runs':{}}
for name,lib in [('before',SNAP),('after',R/'candidate-bin')]:
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(lib)
    p=subprocess.run([str(R/'tail-rollback')],env=env,capture_output=True,text=True)
    (R/f'tail-{name}.jsonl').write_text(p.stdout);(R/f'tail-{name}.stderr.log').write_text(p.stderr)
    result['runs'][name]={'exit_code':p.returncode,'library_directory':str(lib),'libllama_sha256':hashlib.sha256((lib/'libllama.so.0.4.0').read_bytes()).hexdigest(),'summary':json.loads(p.stdout.splitlines()[-1]) if p.stdout else None}
(R/'tail-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['runs'],indent=2))
assert result['runs']['before']['exit_code']==1 and result['runs']['after']['exit_code']==0
