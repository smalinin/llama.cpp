import hashlib,json,subprocess
from pathlib import Path

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp')
SNAP=R.parent/'stage8/candidate-bin'
header=(R/'diagnostic-callback.h').read_text()
old='return options.mode.find("float2d") != std::string::npos ||'
new='''return options.mode.find("float2d") != std::string::npos ||
        (options.mode.find("compressor") != std::string::npos &&
         (name.find("comp_state_kv-") == 0 || name.find("comp_state_score-") == 0)) ||'''
assert header.count(old)==1
(R/'candidate-callback.h').write_text(header.replace(old,new))
source=(R/'capture-attention.cpp').read_text().replace('"diagnostic-callback.h"','"candidate-callback.h"')
old='for (const std::string mode : {"decode-scalar-fa-upgate-hc-router-down"})'
new='for (const std::string mode : {"decode-scalar-fa-upgate-hc-router-down", "decode-scalar-fa-upgate-hc-router-down-compressor", "decode-scalar-fa-upgate-hc-router-down-float2d"})'
assert source.count(old)==1
(R/'compressor-candidate.cpp').write_text(source.replace(old,new))
command=['g++','-std=c++17','-O2',f'-I{REPO}/include',f'-I{REPO}/src',f'-I{REPO}/ggml/include',str(R/'compressor-candidate.cpp'),f'-L{SNAP}',f'-Wl,-rpath,{SNAP}','-lllama','-lggml','-lggml-base','-o',str(R/'compressor-candidate')]
with (R/'candidate-build.txt').open('w') as log:
    proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
manifest={'command':command,'exit_code':proc.returncode,'sources':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['prepare-compressor-candidate.py','diagnostic-callback.h','candidate-callback.h','capture-attention.cpp','compressor-candidate.cpp']}}
(R/'candidate-build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
raise SystemExit(proc.returncode)
