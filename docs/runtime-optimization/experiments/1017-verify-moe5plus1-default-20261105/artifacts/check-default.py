from pathlib import Path
import json,os,subprocess,hashlib
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent
B=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-performance-20261105/artifacts'
O=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts'
cfg=json.loads((B/'config.json').read_text());assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
rows={};proof=[]
for label,value in [('default',None),('on','1'),('off','0'),('off-word','off')]:
 env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LD_LIBRARY_PATH=str(P/'candidate')+':'+str(R/'build-aliceai/bin'),LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0');env.pop('LLAMA_DSV41_VERIFY_MOE_WIDE',None)
 if value is not None:env['LLAMA_DSV41_VERIFY_MOE_WIDE']=value
 cmd=[str(O/'metadata'),str(P/(label+'.json')),'measure',*cfg['command'][1:]]
 with(P/(label+'.log')).open('wb')as log:subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((P/(label+'.json')).read_text());assert d['complete']and d['gpu_complete_bundles']==15360 and d['fits_512mib'];rows[label]=d;proof.append(dict(label=label,value=value,command=cmd,environment={k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','LD_'))}))
keys=['memory','context','gpu_layers','target_rs','target_outputs_per_seq','split','overrides','routes','routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles']
# Ignore changing free-memory readings; compare all reserved allocations and routing.
def allocations(d):return[{k:v for k,v in x.items()if k not in ['free_before','projected_free']}for x in d['memory']]
for a,b in [('default','on'),('off','off-word')]:
 for k in keys:
  if k!='memory':assert rows[a][k]==rows[b][k],(a,b,k)
 assert allocations(rows[a])==allocations(rows[b]),(a,b)
for label,old in [('on','metadata-1.json'),('off','metadata-0.json')]:
 d=json.loads((B/old).read_text());assert allocations(rows[label])==allocations(d)
 for k in keys:
  if k!='memory':assert rows[label][k]==d[k],(label,k)
assert allocations(rows['on'])!=allocations(rows['off'])
(P/'checks.json').write_text(json.dumps(dict(complete=True,default_equals_explicit_one=True,zero_equals_off_word=True,allocations_and_routes_match_prior_measured_candidate=True,all_gpu_bundles=15360,scope='Joint no_alloc metadata guard/default check only; prior actual correctness/performance evidence unchanged. No new universal determinism claim.',runs=proof),indent=2)+'\n')
print('DEFAULT_CHECK_PASS')
