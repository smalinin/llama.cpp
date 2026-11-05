from pathlib import Path
import json,os,subprocess,hashlib
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent
B=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-performance-20261105/artifacts'
O=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts'
cfg=json.loads((P/'config.json').read_text());assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert json.loads((P/'operator-results.json').read_text())['complete']
res={};runs=[]
for flag in ['0','1']:
 env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LD_LIBRARY_PATH=str(P/'candidate')+':'+str(R/'build-aliceai/bin'),LLAMA_DSV41_VERIFY_FLOAT_WIDE=flag);env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None)
 cmd=[str(O/'metadata'),str(P/('metadata-'+flag+'.json')),'measure',*cfg['command'][1:]]
 with(P/('metadata-'+flag+'.log')).open('wb')as log:subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 res[flag]=json.loads((P/('metadata-'+flag+'.json')).read_text());assert res[flag]['complete'];runs.append(dict(flag=flag,command=cmd,environment={k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','LD_'))}))
keys=['routes','routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles','context','gpu_layers','split','overrides'];assert all(res['0'][k]==res['1'][k]for k in keys);assert res['0']['gpu_complete_bundles']==15360
old=json.loads((B/'metadata-1.json').read_text());assert all(res['0'][k]==old[k]for k in keys)
actual=json.loads((B/'native-results.json').read_text())['arms'][1]['memory'];uu={v['index']:k for k,v in actual['inference'].items()if isinstance(v,dict)};rows=[]
for v in res['1']['memory']:
 idx=int(v['device'][4:]);ref=next(x for x in old['memory']if x['device']==v['device']);prior=min(actual[ph][uu[idx]]['min_free']for ph in ['load','inference']);over=max(0,ref['projected_free']-prior);free=v['projected_free']-over-64*2**20;rows.append(dict(device=v['device'],conservative_free_mib=free/2**20,prior_overhead_mib=over/2**20,admit=free>=512*2**20))
j=dict(complete=True,admit=all(x['admit']for x in rows),rows=rows,placement_exact=True,gpu_complete_bundles=15360,runs=runs,method='Fresh joint metadata, minus prior same-context actual overhead and64MiB. Rebuilt module identities recorded separately; monitored actual memory remains mandatory.')
(P/'admission.json').write_text(json.dumps(j,indent=2)+'\n');assert j['admit'];print('FLOAT6_ADMITTED',min(x['conservative_free_mib']for x in rows))
