#!/usr/bin/env python3
"""Sequential prompt-cache comparisons using the existing server driver."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, resource, subprocess, time
R=Path(__file__).resolve().parent; PREV=R.parent/'stage28'; REPO=Path('/home/sergei/Github/llama.cpp'); LIB=PREV/'final-bin'
spec=importlib.util.spec_from_file_location('stage2',R.parent/'stage2/run-parallel.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
GPU=json.loads((R.parent/'stage9/current-server/off/manifest.json').read_text())['gpu_order']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def requests(port,directory,result,api):
 prompts=[json.loads((PREV/'server-runs/after-unified/glm5next-spec-1'/f'serial-{i}-request.json').read_text())['prompt'] for i in range(3)]
 prompts.append('Read the following numbered facts.\n'+''.join(f'Entry {i}: the checkpoint value is {1000+i}; the label is blue.\n' for i in range(24))+'\nExplain how to check these entries, then list the first ten checkpoint values.\nAnswer:\n')
 def request(label,index,cache):
  payload={'prompt':prompts[index],'n_predict':64,'temperature':0,'seed':1234,'top_k':40,'top_p':.95,'min_p':.05,'cache_prompt':cache,'return_tokens':True,'id_slot':index%3,'n_probs':5,'post_sampling_probs':False}
  save(directory/f'{label}-{index}-request.json',payload);start=time.monotonic()
  response=api(port,'/completion',payload);save(directory/f'{label}-{index}-response.json',response)
  if 'error' in response or not response.get('tokens'):raise RuntimeError(str(response))
  item={'phase':label,'prompt':index,'slot':index%3,'wall_s':time.monotonic()-start,'timings':response.get('timings'),'tokens':len(response['tokens'])}
  result['requests'].append(item);save(directory/'result.json',result)
  print('REQUEST',label,index,json.dumps(item['timings']),flush=True)
 for phase,cache in [('fresh1',False),('reuse1',True),('reuse2',True),('fresh2',False)]:
  for i in range(3):request(phase,i,cache)
 for phase,cache in [('fresh1',False),('reuse1',True),('reuse2',True),('fresh2',False)]:request(phase,3,cache)
def run(label,kind,cache):
 b=json.loads((PREV/'final-build-manifest.json').read_text());hashes=b['candidate_hashes']
 assert all(sha(LIB/n)==h for n,h in hashes.items())
 assert all(sha(REPO/n)==h for n,h in b['sources'].items())
 code=base.source.replace("'-fit', 'on', '--fit-target', '3072',", "'-fit', 'off', '-ngl', '99', '--tensor-split', '1,1,1,1,1,0.4',")
 code=code.replace("'--metrics',", "'--metrics', '--kv-unified', '--cache-ram', '2048', '--cache-idle-slots', '--verbose',")
 code=code.replace("env = os.environ.copy()", "env = os.environ.copy()\n    for key in list(env):\n        if key.startswith('LLAMA_GLM5_') or key in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','NVIDIA_TF32_OVERRIDE'): env.pop(key)\n    env['LLAMA_GLM5_POOL_CACHE'] = "+repr(str(cache)))
 code=code.replace("            save(directory / 'props.json'", "            print('READY',round(result['startup_s'],1),flush=True)\n            save(directory / 'props.json'")
 start=code.index('            from concurrent.futures import ThreadPoolExecutor');end=code.index('        except Exception as error:',start)
 code=code[:start]+"            run_requests(port,directory,result,api)\n            result['status']='passed'\n"+code[end:]
 ns={'__name__':'stage29_server','run_requests':requests};exec(compile(code,str(R.parent/'stage2/run-parallel.py'),'exec'),ns);ns['GPU_ORDER']=GPU
 profile=ns['PROFILES']['glm5next'];profile['spec_type']=kind
 if kind=='draft-dflash':profile['draft']=Path('/home/sergei/.models/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/GLM-5.3-Flash-DFlash2-Q8_0.gguf')
 out=R/'server-runs'/label;out.mkdir(parents=True,exist_ok=False)
 inputs=list(profile['model'].parent.glob('GLM-5.3-Flash-UD-Q4_K_XL-*-of-*.gguf'))+([profile['draft']] if 'draft' in profile else [])
 meta={'label':label,'library_hashes':hashes,'source_hashes':b['sources'],'pool_cache':cache,'request_schedule':'fully sequential, fixed slots; fresh1/reuse1/reuse2/fresh2','models':{str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in inputs},'driver_sources':{str(p):sha(p) for p in [Path(__file__),R.parent/'stage2/run-parallel.py',R.parent/'run_baseline.py']}}
 save(out/'control-manifest.json',meta);print('CONTROL',label,flush=True)
 result=ns['run_case']('glm5next',kind!='none',out,LIB/'llama-server')
 assert all(Path(p).stat().st_size==v['size'] and Path(p).stat().st_mtime_ns==v['mtime_ns'] for p,v in meta['models'].items())
 return result
if __name__=='__main__':
 cases={'dflash-on':('draft-dflash',1),'dflash-off':('draft-dflash',0),'mtp-on':('draft-mtp',1),'mtp-off':('draft-mtp',0),'native-on':('none',1),'native-off':('none',0)}
 p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',choices=cases,default=['dflash-on','dflash-off','mtp-on','mtp-off']);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 for label in a.cases:
  result=run(label,*cases[label])
  if result['status']!='passed':raise SystemExit(1)
