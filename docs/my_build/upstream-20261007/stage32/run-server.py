import argparse, hashlib, importlib.util, json, os, socket, subprocess, time, urllib.error
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('base',R.parent/'run_baseline.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('bin');p.add_argument('--long',action='store_true');p.add_argument('--graphs',action='store_true');a=p.parse_args()
out=R/a.label;out.mkdir(exist_ok=False);(out/'slots').mkdir()
lib=Path(a.bin).resolve()
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
cmd=[str(lib/'llama-server'),'-m',str(base.PROFILES['deepseek41']['model']),'-c','8192','-b','2048','-ub','512','-np','1','-ctk','f16','-ctv','f16','-t','12','-tb','12','-fit','off','-ngl','99','--tensor-split','1,1,1,1,1,0.4','--split-mode','layer','--load-mode','none','--lazy-mode','auto','--host','127.0.0.1','--port',str(port),'--slot-save-path',str(out/'slots')+'/','--jinja']
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_','LLAMA_DSV41_')):env.pop(k)
env.update(LD_LIBRARY_PATH=str(lib),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='GPU-3972c9ef-dd88-6b95-ac0c-fc9b0eb86f76,GPU-9e5fb796-93b1-b72d-1fd8-438d93aac573,GPU-8e5bef34-d86f-b6ed-3419-3d0c1be7dd11,GPU-7f297c35-ddfe-c2dd-9d94-fbf1477b86b6,GPU-dd8b5c70-2afd-3c20-7b36-59469a2b411a,GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783',GGML_CUDA_DISABLE_GRAPHS='1')
if a.graphs:env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
def sha(f):
 with Path(f).open('rb') as g:return hashlib.file_digest(g,'sha256').hexdigest()
manifest={'command':cmd,'env':{k:env.get(k) for k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_GRAPHS']},'hashes':{f.name:sha(f) for f in lib.iterdir() if f.is_file()}}
base.save(out/'manifest.json',manifest)
def call(label,path,req):
 base.save(out/(label+'-request.json'),req);t=time.monotonic();res=base.api(port,path,req,timeout=900);base.save(out/(label+'-response.json'),res)
 assert 'error' not in res,res
 print(label,round(time.monotonic()-t,2),res.get('timings',{}),flush=True);return res
def complete(label,prompt,cached=True,n=32,probs=5):
 return call(label,'/completion',{'prompt':prompt,'n_predict':n,'temperature':0,'seed':1234,'top_k':40,'top_p':.95,'min_p':.05,'cache_prompt':cached,'return_tokens':True,'stream':False,'n_probs':probs,'ignore_eos':True})
def comparison(x,y):
 return {'tokens_equal':x['tokens']==y['tokens'],'first_token_difference':next((i for i,(u,v) in enumerate(zip(x['tokens'],y['tokens'])) if u!=v),None),'probabilities_equal':(x.get('completion_probabilities')==y.get('completion_probabilities')) if x.get('completion_probabilities') else None,'control_timings':x['timings'],'restored_timings':y['timings']}
results={}
with (out/'server.log').open('w') as log:
 proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+900
  while True:
   assert proc.poll() is None,'server exited'
   try:
    if base.api(port,'/health',timeout=3).get('status')=='ok':break
   except (urllib.error.URLError,TimeoutError):pass
   assert time.monotonic()<deadline,'startup timeout'
   time.sleep(1)
  maps=Path(f'/proc/{proc.pid}/maps').read_text().splitlines();libs=sorted({x.split()[-1] for x in maps if '/libllama' in x or '/libggml' in x});assert all(Path(f).parent==lib for f in libs)
  manifest['loaded_libraries']={f:sha(f) for f in libs};base.save(out/'manifest.json',manifest)
  initial=complete('initial',base.PROMPT,False)
  tok=call('tokenize','/tokenize',{'content':base.PROMPT,'add_special':True,'parse_special':True})['tokens'];prefix=tok+initial['tokens']
  call('save','/slots/0?action=save',{'filename':'prefix.bin'})
  control=complete('control',prefix)
  call('erase','/slots/0?action=erase',{})
  call('restore','/slots/0?action=restore',{'filename':'prefix.bin'})
  restored=complete('restored',prefix)
  assert control['timings']['cache_n']==restored['timings']['cache_n']==len(prefix)-1
  results['session']=comparison(control,restored);base.save(out/'results.json',results);print(results,flush=True)
  if a.long:
   prompt=json.loads((R.parent/'stage25/free-runs/snapshot-off/ratio2-long-fresh-request.json').read_text())['prompt']
   fresh=complete('long-fresh',prompt,False,1024,0);cached=complete('long-cache',prompt,True,1024,0)
   assert cached['timings']['cache_n']>6000
   results['long']=comparison(fresh,cached);base.save(out/'results.json',results);print(results,flush=True)
 finally:
  if proc.poll() is None:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
