import json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');B=REPO/'build-glm53'
e=next(x for x in json.loads((B/'compile_commands.json').read_text()) if x['file']==str(REPO/'tests/test-backend-ops.cpp'))
c=shlex.split(e['command']);obj=R/'test-backend-ops.cpp.o';c[c.index('-o')+1]=str(obj);subprocess.run(c,cwd=e['directory'],check=True)
c=shlex.split((B/'tests/CMakeFiles/test-backend-ops.dir/link.txt').read_text())
for i,v in enumerate(c):
 if v.endswith('.o'):c[i]=str(obj) if v.endswith('test-backend-ops.cpp.o') else str((B/'tests'/v).resolve())
 elif '.so' in v and not v.startswith('-'):c[i]=str(R/'candidate-bin'/Path(v).name)
 elif v.startswith('-Wl,-rpath,'):c[i]='-Wl,-rpath,'+str(R/'candidate-bin')
c[c.index('-o')+1]=str(R/'test-backend-ops');subprocess.run(c,cwd=B/'tests',check=True)
