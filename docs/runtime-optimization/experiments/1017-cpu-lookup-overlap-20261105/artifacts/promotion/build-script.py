from pathlib import Path
import json,hashlib,subprocess
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts';D=P/'promotion';D.mkdir();C=D/'candidate';C.mkdir()
s=R/'ggml/src/ggml-backend.cpp';t=s.read_text();old='sched->cpu_lookup_early = early_cpu && strcmp(early_cpu, "1") == 0;';assert t.count(old)==1;s.write_text(t.replace(old,'sched->cpu_lookup_early = !early_cpu || strcmp(early_cpu, "1") == 0;'))
h=(P/'scheduler.cpp').read_text();old='if(mode!=1||!(v=="eligible"||v=="view"||v=="shapes"||v=="switch"||v=="failure"))';assert old in h;h=h.replace(old,'if(mode==0||!(v=="eligible"||v=="view"||v=="shapes"||v=="switch"||v=="failure"))');(D/'scheduler.cpp').write_text(h)
b=json.loads((P/'final-build.json').read_text());cc=[x.replace(str(P/'candidate'),str(C))for x in b['compile']];link=[x.replace(str(P/'candidate'),str(C))for x in b['link']]
compile_harness=['/usr/bin/c++','-O2','-std=c++17','-I'+str(R/'ggml/include'),'-I'+str(R/'ggml/src'),'-I'+str(R/'vendor'),'-I/usr/local/cuda/include',str(D/'scheduler.cpp'),'-L'+str(R/'build-aliceai/bin'),'-L/usr/local/cuda/lib64','-Wl,-rpath,'+str(R/'build-aliceai/bin'),'-lggml','-lggml-base','-lggml-cpu','-lggml-cuda','-lcudart','-o',str(D/'scheduler-harness')]
with(D/'build.log').open('w')as log:
 for cmd in [cc,link,compile_harness]:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(C/'libggml-base.so').symlink_to('libggml-base.so.0')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(D/'build.json').write_text(json.dumps({'compile':cc,'link':link,'harness_compile':compile_harness,'source_sha256':sha(s),'candidate_sha256':sha(C/'libggml-base.so.0'),'old_measured_source_sha256':b['source_sha256'],'old_measured_library_sha256':b['candidate_sha256'],'change':'Startup default only: unset and 1 ON, all other values OFF; isolation guards unchanged.'},indent=2)+'\n')
print('Default changed and private promotion library/harness built.')
