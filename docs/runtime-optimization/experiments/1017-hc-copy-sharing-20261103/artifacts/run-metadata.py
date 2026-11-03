from pathlib import Path
import json,subprocess,os,hashlib
P=Path(__file__).parent
R=Path('/home/sergei/Github/llama.cpp')
cfg=json.load(open('/tmp/dsv41-m2-large-context-20261103/config.json'))
cfg['source_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
cfg['runtime_patch']=subprocess.check_output(['git','diff','--','ggml/src/ggml-backend.cpp'],cwd=R,text=True)
cfg['source_library_hashes']={n:hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()for n in cfg['source_library_hashes']}
(P/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']}
env.update(cfg['environment'])
args=cfg['command'][1:]
out={'complete':False,'cases':[],'helper_sha256':hashlib.file_digest((P/'metadata').open('rb'),'sha256').hexdigest()}
for ctx,mode in [(100352,'default'),(100352,'off'),(100352,'on'),(176128,'on'),(100352,'k5-off'),(100352,'k5-on')]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
 e=env.copy()
 if mode!='default':e['GGML_SCHED_COPY_RESHAPE']='1'if mode.endswith('on')else'0'
 cmd=[str(P/'metadata'),str(P/f'metadata-{ctx}-{mode}.json'),'measure',*args]
 cmd[cmd.index('--ctx-size')+1]=str(ctx)
 if mode.startswith('k5'):
  cmd[cmd.index('--spec-draft-n-max')+1]='5'
  cmd += ['--no-spec-draft-backend-sampling']
 with (P/f'metadata-{ctx}-{mode}.log').open('wb')as log:
  rc=subprocess.run(cmd,cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
 assert rc==0
 j=json.load(open(cmd[1]));assert j['complete']and j['fits_512mib']and j['gpu_complete_bundles']==15360 and j['routed_host_bytes']==0
 out['cases'].append({'context':ctx,'mode':mode,'command':cmd,'environment':{k:v for k,v in e.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))},'result':j})
 (P/'metadata-results.json').write_text(json.dumps(out,indent=2)+'\n')
 print('META',ctx,mode,[(x['device'],round(x['main_compute']/1048576,2),round(x['projected_free']/1048576,2))for x in j['memory']],flush=True)
a,b,c=[x['result']for x in out['cases'][:3]]
for key in ['routes','context','routed_gpu_bytes','routed_host_bytes']:
 assert a[key]==b[key]==c[key]
for x,y,z in zip(a['memory'],b['memory'],c['memory']):
 for key in ['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:
  assert x[key]==y[key],('default/off',key)
 for key in ['main_model','main_context','draft_and_taps_model','draft_and_taps_context']:
  assert x[key]==z[key],('unchanged budget',key)
out['complete']=True
(P/'metadata-results.json').write_text(json.dumps(out,indent=2)+'\n')
