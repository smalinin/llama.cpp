from pathlib import Path
import json,os,subprocess,hashlib
r=Path(__file__).resolve().parent;lib=r/'final-bin';out=r/'final-regression';out.mkdir(exist_ok=True);env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(lib),CUDA_VISIBLE_DEVICES='');cases=json.loads((r.parent/'stage32/final-regression.json').read_text())['cases'];results=[]
for c in cases:
 c=dict(c);c['command']=[x.replace('/stage32/final-regression/glm-layout','/stage33/final-regression/glm-layout') for x in c['command']]
 with (out/(c['label']+'.log')).open('w') as log:p=subprocess.run(c['command'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
 c['exit_code']=p.returncode;results.append(c);print(c['label'],p.returncode,flush=True)
 (r/'final-regression.json').write_text(json.dumps({'library_sha256':hashlib.sha256((lib/'libllama.so.0.4.0').read_bytes()).hexdigest(),'cases':results},indent=2)+'\n');assert p.returncode==0
