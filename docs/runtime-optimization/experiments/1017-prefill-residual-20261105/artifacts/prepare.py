from pathlib import Path
import subprocess,json,hashlib,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105';P=E/'artifacts';P.mkdir(parents=True)
(E/'.gitignore').write_text('artifacts/*.so\nartifacts/__pycache__/\n!*.log\n');shutil.copy2('/tmp/dsv41-residual-probe.cpp',P/'probe.cpp');shutil.copy2(R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts/memory_monitor.py',P/'memory_monitor.py')
argv=['/usr/bin/c++','-O2','-std=c++17','-fPIC','-shared',str(P/'probe.cpp'),*[ '-I'+str(R/x)for x in ['src','include','common','ggml/src','ggml/include','vendor']],'-ldl','-pthread','-o',str(P/'probe.so')]
with(P/'probe-build.log').open('w')as log:subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();(P/'probe-build.json').write_text(json.dumps({'command':argv,'source_sha256':sha(P/'probe.cpp'),'library_sha256':sha(P/'probe.so'),'kind':'External host API timing only; no CUDA API interposition, CUPTI, added GPU events, callbacks or changed math. Wrapping existing backend copy interface only forwards original return/status and timing.'},indent=2)+'\n');print('Residual probe built.')
