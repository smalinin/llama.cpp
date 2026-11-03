from pathlib import Path
import os,json,subprocess,time,hashlib,ctypes,csv,re
P=Path(__file__).resolve().parent;R=Path('/home/sergei/Github/llama.cpp');c=json.load(open(P/'config.json'));env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(c['environment']);args=c['command'][1:]
for k in ['LLAMA_DSV41_PREFILL_SKIP','LLAMA_DSPARK_GPU_FEATURES','LLAMA_DSPARK_GPU_FEATURES_LOCAL','LLAMA_DSPARK_GPU_FEATURES_ASYNC']:env.pop(k,None)
r={'complete':False,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'cases':[],'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'helper_sha256':hashlib.sha256((P/'metadata').read_bytes()).hexdigest()}
for ctx,mode in [(100352,'default'),(176128,'default'),(176128,'off'),(163840,'default'),(159744,'default')]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
 e=env.copy()
 if mode=='off':e['LLAMA_DSV41_PREFILL_SKIP']='0'
 cmd=[str(P/'metadata'),str(P/f'metadata-{ctx}-{mode}.json'),'measure',*args];cmd[cmd.index('--ctx-size')+1]=str(ctx);start=time.monotonic();print('METADATA_START',ctx,mode,flush=True)
 with(P/f'metadata-{ctx}-{mode}.log').open('wb')as f:rc=subprocess.run(cmd,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 v=json.load(open(cmd[1]));assert rc==0 and v['complete']and v['gpu_complete_bundles']==15360 and v['routed_host_bytes']==0,v.get('error');r['cases'].append({'context':ctx,'mode':mode,'command':cmd,'exit_code':rc,'seconds':time.monotonic()-start,'result':v});(P/'metadata-results.json').write_text(json.dumps(r,indent=2)+'\n');print('METADATA_DONE',ctx,mode,[(x['device'],round(x['projected_free']/1048576,2))for x in v['memory']],flush=True)
a=r['cases'][1]['result'];b=r['cases'][2]['result']
for key in ['routes','context','target_rs','target_outputs_per_seq','routed_gpu_bytes','routed_host_bytes']:assert a[key]==b[key],key
for x,y in zip(a['memory'],b['memory']):
 for k in ['device','total','main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k],(k,x[k],y[k])
r['default_off_budget_exact']=True;r['complete']=True;(P/'metadata-results.json').write_text(json.dumps(r,indent=2)+'\n');print('METADATA_COMPLETE',flush=True)
