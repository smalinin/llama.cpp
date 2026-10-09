from pathlib import Path
import argparse,hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();results=[]
for build in ['before','after']:
 lib=R.parent/'stage30/preclear-bin' if build=='before' else R/'candidate-bin'
 for q8 in [0,1]:
  out=R/'layout-restore'/f'{build}-gpu{a.gpu}-q{q8}';out.mkdir(parents=True,exist_ok=False)
  cmd=[str(R/'restore-layout'),'/home/sergei/Github/llama.cpp/build-glm53/tests/test-models/glm5next-moe.gguf',str(out),str(a.gpu),str(q8)];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783' if a.gpu else '',LD_LIBRARY_PATH=str(lib),LLAMA_GLM5_POOL_CACHE='0')
  with (out/'run.log').open('w') as err:p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=err,text=True)
  (out/'results.jsonl').write_text(p.stdout);rows=[json.loads(s) for s in p.stdout.splitlines() if s.startswith('{')];print(build,'GPU',a.gpu,'Q8',q8,p.returncode,rows,flush=True)
  with (lib/'libllama.so.0.4.0').open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  results.append({'build':build,'gpu':a.gpu,'q8':q8,'command':cmd,'library_sha256':h,'exit_code':p.returncode,'rows':rows});(R/f'layout-restore-gpu{a.gpu}.json').write_text(json.dumps(results,indent=2)+'\n')
  assert p.returncode==0
  if build=='after':assert len(rows)==5 and all(x['logits_equal'] for x in rows)
