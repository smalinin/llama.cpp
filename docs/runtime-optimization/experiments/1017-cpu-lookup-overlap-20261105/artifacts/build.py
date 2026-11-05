from pathlib import Path
import subprocess,shlex,hashlib,json,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts';B=R/'build-aliceai/ggml/src';D=E/'candidate';D.mkdir(exist_ok=True)
# Keep production libraries intact; only the scheduler translation unit changes.
flags={}
for l in(B/'CMakeFiles/ggml-base.dir/flags.make').read_text().splitlines():
 if l.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):k,v=l.split('=',1);flags[k.strip()]=shlex.split(v.strip())
cc=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(R/'ggml/src/ggml-backend.cpp'),'-o',str(D/'ggml-backend.cpp.o')]
argv=shlex.split((B/'CMakeFiles/ggml-base.dir/link.txt').read_text());idx=argv.index('-o');argv[idx+1]=str(D/'libggml-base.so.0')
for i,s in enumerate(argv):
 if s.endswith('.o'):argv[i]=str(D/'ggml-backend.cpp.o')if s.endswith('/ggml-backend.cpp.o')else str(B/s)
with(E/'candidate-build.log').open('w')as log:
 subprocess.run(cc,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True);subprocess.run(argv,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
(D/'libggml-base.so').unlink(missing_ok=True);(D/'libggml-base.so').symlink_to('libggml-base.so.0')
for n in ['baseline']:
 d=E/n;d.mkdir(exist_ok=True);shutil.copy2(R/'build-aliceai/bin/libggml-base.so.0',d/'libggml-base.so.0');(d/'libggml-base.so').unlink(missing_ok=True);(d/'libggml-base.so').symlink_to('libggml-base.so.0')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
(E/'candidate-build.json').write_text(json.dumps({'compile':cc,'link':argv,'source_sha256':sha(R/'ggml/src/ggml-backend.cpp'),'candidate_sha256':sha(D/'libggml-base.so.0'),'baseline_sha256':sha(E/'baseline/libggml-base.so.0'),'production_base_sha256':sha(R/'build-aliceai/bin/libggml-base.so.0')},indent=2)+'\n');print('isolated candidate built')
