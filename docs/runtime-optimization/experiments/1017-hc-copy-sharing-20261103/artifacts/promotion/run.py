from pathlib import Path
import json,subprocess,os,hashlib
P=Path(__file__).parent;O=P.parent;R=Path('/home/sergei/Github/llama.cpp')
cfg=json.load(open(O/'config.json'));env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(cfg['environment'])
args=cfg['command'][1:]
j={'complete':False,'scope':'Default selector promotion only;explicitON full correctness/speed evidence in parent unchanged. New default vs explicit1 and OFF metadata/operator controls.','source_patch':subprocess.check_output(['git','diff','--','ggml/src/ggml-backend.cpp'],cwd=R,text=True),'library_hashes':{n:hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()for n in cfg['source_library_hashes']},'cases':[]}
for mode,value in [('default',None),('on','1'),('off','0')]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
 e=env.copy()
 if value is not None:e['GGML_SCHED_COPY_RESHAPE']=value
 else:e.pop('GGML_SCHED_COPY_RESHAPE',None)
 cmd=[str(O/'metadata'),str(P/f'metadata-{mode}.json'),'measure',*args]
 with(P/f'metadata-{mode}.log').open('wb')as f:rc=subprocess.run(cmd,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 assert rc==0
 v=json.load(open(P/f'metadata-{mode}.json'));assert v['complete']and v['fits_512mib']
 j['cases'].append({'mode':mode,'command':cmd,'result':v});(P/'results.json').write_text(json.dumps(j,indent=2)+'\n');print('PROM_META',mode,flush=True)
a,b,c=[x['result']for x in j['cases']]
for key in ['routes','context','routed_gpu_bytes','routed_host_bytes']:assert a[key]==b[key]==c[key]
for x,y in zip(a['memory'],b['memory']):
 for key in ['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[key]==y[key],key
j['default_explicit1_budget_exact']=True;j['complete']=True;(P/'results.json').write_text(json.dumps(j,indent=2)+'\n');(P/'PROMOTED.patch').write_text(j['source_patch']);print('PROM_METADATA_COMPLETE',flush=True)
