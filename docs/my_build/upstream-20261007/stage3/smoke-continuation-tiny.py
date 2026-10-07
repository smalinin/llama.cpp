import json,os,socket,subprocess,time,urllib.request
from pathlib import Path
S=Path('/home/sergei/_my_sync/llama_upstream_review/stage3')
directory=S/'continuation-tiny-smoke';directory.mkdir(exist_ok=False);(directory/'slots').mkdir()
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['LD_LIBRARY_PATH']=str(S/'bin')
command=[str(S/'bin/llama-server'),'-m','/home/sergei/Github/llama.cpp/build-glm53/tinyllamas/stories15M-q4_0.gguf','-c','256','-np','1','-ngl','0','-t','2','-tb','2','--host','127.0.0.1','--port',str(port),'--slot-save-path',str(directory/'slots')+'/']
def api(port,path,data=None):
 request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
with (directory/'server.log').open('w') as log:
 p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   try:
    if api(port,'/health')['status']=='ok':break
   except Exception:time.sleep(.1)
  prompt='Once upon a time'
  response=api(port,'/completion',{'prompt':prompt,'n_predict':8,'temperature':0,'return_tokens':True,'cache_prompt':False})
  (directory/'state-old-version-response.json').write_text(json.dumps(response,indent=2)+'\n')
  n={};exec((S/'check-continuation.py').read_text(),n);result={}
  n['check_continuation'](api,port,directory,result,prompt)
  print(json.dumps(result))
 finally:
  p.terminate();p.wait(timeout=30)
