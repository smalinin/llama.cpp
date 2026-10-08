#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
spec=importlib.util.spec_from_file_location('baseline',ROOT.parent/'run_baseline.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SNAP=ROOT.parent/'stage8/candidate-bin';EXPERIMENT=ROOT/'experiment-bin'
GPU_ORDER=json.loads((ROOT.parent/'stage9/current-server/off/manifest.json').read_text())['gpu_order']
QUESTIONS={'svg':'Create a complete HTML page with an SVG of a pelican riding a bicycle. Use no external assets.','python':'Write a Python function that merges overlapping intervals. Include type hints and three examples.','explain':'Explain why a database index speeds up reads but can slow down writes. Give a concrete example.'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(config):
 directory=ROOT/'free-runs'/config;directory.mkdir(parents=True,exist_ok=False)
 profile=base.PROFILES['deepseek41'];snapshot=config.startswith('snapshot-');enabled=config.startswith('candidate-');n=1 if config.endswith('n1') else 3 if config.endswith('n3') else 0
 server=(SNAP if snapshot else EXPERIMENT)/'llama-server'
 with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
 command=[str(server),'-m',str(profile['model']),'-c','8192','-b','2048','-ub','512','-np','1','-ctk','f16','-ctv','f16','-ctkd','f16','-ctvd','f16','-t','12','-tb','12','-fit','off','-ngl','99','--tensor-split','1,1,1,1,1,0.4','--split-mode','layer','--load-mode','none','--lazy-mode','auto','--host','127.0.0.1','--port',str(port),'--metrics','--jinja','--verbosity','3']
 if n:command+=['--spec-type','draft-dspark','--spec-draft-n-max',str(n),'--spec-draft-p-min','0','--model-draft',str(profile['draft']),'--spec-draft-ngl','99']
 env=os.environ.copy()
 for k in list(env):
  if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
 for k in ['GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','LLAMA_DSV41_DIAGNOSTIC']:env.pop(k,None)
 env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=GPU_ORDER,LD_LIBRARY_PATH=str(server.parent),GGML_CUDA_DISABLE_GRAPHS='1',LLAMA_DSV41_DIAGNOSTIC='compressor-down' if enabled else '0')
 hashes=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text()) if snapshot else json.loads((ROOT/'build-manifest.json').read_text())['experiment_hashes']
 assert all(sha(server.parent/k)==v for k,v in hashes.items())
 manifest={'config':config,'command':command,'gpu_order':GPU_ORDER,'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'binary_hashes':hashes,'library_directory':str(server.parent),'candidate_enabled':enabled,'disable_graphs':True,'output_capacity':None if snapshot else 4,'draft_callback':False,'production_change':False,'scope':'Free greedy correctness. Callback recomputes after original ops; timing includes diagnostic overhead and is not a production optimization benchmark.','requests':[]}
 base.save(directory/'manifest.json',manifest);result={'config':config,'status':'running','requests':[]}
 print('START',config,flush=True);started=time.monotonic();stop=threading.Event();peaks={'started':started,'gpu_mib':{},'process_hwm_kib':0}
 with (directory/'server.log').open('w') as log:
  proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
  watcher=threading.Thread(target=base.monitor,args=(proc,directory,stop,peaks),daemon=True);watcher.start()
  try:
   while True:
    if proc.poll() is not None:raise RuntimeError('startup exit '+str(proc.returncode))
    try:
     if base.api(port,'/health',timeout=3).get('status')=='ok':break
    except (urllib.error.URLError,TimeoutError):pass
    if time.monotonic()-started>900:raise TimeoutError('startup timeout')
    time.sleep(1)
   maps=Path(f'/proc/{proc.pid}/maps').read_text().splitlines();libs=sorted({line.split()[-1] for line in maps if any('/'+p in line for p in ['libllama','libggml','libmtmd'])})
   assert libs and all(Path(p).parent.resolve()==server.parent.resolve() for p in libs),libs
   manifest['loaded_libraries']={p:sha(p) for p in libs};assert all(hashes[Path(p).name]==v for p,v in manifest['loaded_libraries'].items())
   base.save(directory/'props.json',base.api(port,'/props'));manifest['startup_s']=time.monotonic()-started;base.save(directory/'manifest.json',manifest)
   print('READY',config,round(manifest['startup_s'],1),flush=True)
   prompts={'baseline':base.PROMPT}
   for name,text in QUESTIONS.items():
    payload={'messages':[{'role':'user','content':text}],'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
    response=base.api(port,'/apply-template',payload);assert isinstance(response['prompt'],str)
    prompts[name]=response['prompt'];base.save(directory/f'{name}-template-request.json',payload);(directory/f'{name}-prompt.txt').write_text(response['prompt'])
   for repeat in [1,2]:
    for name in (['baseline'] if config=='snapshot-n3' else prompts):
     label=f'{name}-greedy-{repeat}'
     payload={'prompt':prompts[name],'n_predict':256,'temperature':0,'seed':1234,'top_k':40,'top_p':0.95,'min_p':0.05,'cache_prompt':False,'return_tokens':True,'stream':False}
     if repeat==1:payload['n_probs']=5
     base.save(directory/f'{label}-request.json',payload);offset=(directory/'server.log').stat().st_size;now=time.monotonic()
     response=base.api(port,'/completion',payload,timeout=900);elapsed=time.monotonic()-now
     base.save(directory/f'{label}-response.json',response)
     assert 'error' not in response and response.get('tokens') and len(response['tokens'])==response['tokens_predicted'] and response['timings'].get('cache_n',0)==0
     item={'request':label,'prompt':name,'repeat':repeat,'n_probs':5 if repeat==1 else 0,'wall_s':elapsed,'timings':response['timings'],'tokens_predicted':response['tokens_predicted'],'tokens_sha256':hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest(),'content_sha256':hashlib.sha256(response['content'].encode()).hexdigest(),'stop_type':response.get('stop_type'),'log_start_offset':offset,'log_end_offset':(directory/'server.log').stat().st_size}
     result['requests'].append(item);base.save(directory/'result.json',result)
     print('DONE',config,label,'tokens',item['tokens_predicted'],'time',round(elapsed,2),flush=True)
   result['status']='passed'
  except Exception as error:result.update(status='failed',error=repr(error))
  finally:
   stop.set();watcher.join(timeout=10)
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   result.update(server_exit_code=proc.returncode,total_s=time.monotonic()-started,peak_gpu_mib=peaks['gpu_mib'],peak_rss_kib=peaks['process_hwm_kib']);base.save(directory/'result.json',result)
 print('FINISH',config,result['status'],result.get('error',''),flush=True)
 if result['status']!='passed':raise RuntimeError(result)
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--configs',nargs='+',choices=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3'],default=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']);args=parser.parse_args()
 results=[]
 for config in args.configs:
  results.append(run(config));base.save(ROOT/'free-runs/results.json',results)
