from pathlib import Path
import subprocess,os,json,hashlib,argparse
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--gpu',default=0,type=int);p.add_argument('--label',default='fixture');a=p.parse_args();out=R/(a.label+'-gpu'+str(a.gpu));out.mkdir(exist_ok=False);rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for q8 in [0,1]:
 for fa in ([1] if a.gpu or q8 else [0,1]):
  for label,lib in [('before',R.parent/'stage31/candidate-bin'),('after',R/'candidate-bin')]:
   name=f'{label}-fa{fa}-q{q8}';d=out/name;cmd=[str(R/'restore'),str(R/'models/deepseek41-moe.gguf'),str(d),str(a.gpu),str(fa),str(q8)];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783' if a.gpu else '',LD_LIBRARY_PATH=str(lib))
   with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=log,text=True)
   (out/(name+'.jsonl')).write_text(r.stdout);result={'name':name,'command':cmd,'library_sha256':sha(lib/'libllama.so.0.4.0'),'model_sha256':sha(R/'models/deepseek41-moe.gguf'),'exit_code':r.returncode,'rows':[json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]};rows.append(result);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(result['name'],r.returncode,result['rows'],flush=True)
