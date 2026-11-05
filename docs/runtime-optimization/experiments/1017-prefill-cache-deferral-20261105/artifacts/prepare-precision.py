from pathlib import Path
import json,subprocess,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts'
s=(O/'precision.cpp').read_text().replace('for(int width:{1,7,511,513})','for(int width:{1,7,508,511,512,513})');assert 'for(int width:{1,7,508,511,512,513})'in s;(P/'precision.cpp').write_text(s);shutil.copy2(O/'precision-inputs.json',P/'precision-inputs.json');shutil.copy2(O/'fault.cpp',P/'fault.cpp')
base=['/usr/bin/c++','-O2','-std=c++17',*['-I'+str(R/x)for x in ['src','include','common','ggml/include','ggml/src','vendor']]]
cmd=base+[str(P/'precision.cpp'),'-L'+str(R/'build-aliceai/bin'),'-Wl,-rpath,'+str(R/'build-aliceai/bin'),'-lllama-common','-lllama','-lggml','-lggml-base','-lggml-cpu','-lggml-cuda','-lcrypto','-ldl','-pthread','-o',str(P/'precision-harness')]
with(P/'precision-build.log').open('wb')as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
cmd2=base+['-fPIC','-shared',str(P/'fault.cpp'),'-ldl','-pthread','-o',str(P/'fault.so')]
with(P/'fault-build.log').open('wb')as f:subprocess.run(cmd2,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
(P/'precision-build.json').write_text(json.dumps({'command':cmd,'fault_command':cmd2,'note':'Adapted existing external full-model precision harness. Adds508/512 append widths, same text/code98K, full logits/taps/KV/state/restore/rollback/NLL and fault/retry checks. No eval callback.'},indent=2)+'\n');shutil.copy2('/tmp/prepare-deferral-precision.py',P/'prepare-precision.py');print('Precision and fault harness built.')
