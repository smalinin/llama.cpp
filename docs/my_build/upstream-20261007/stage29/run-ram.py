#!/usr/bin/env python3
"""Exercise actual RAM-cache restore with sequential automatic slot assignment."""
from pathlib import Path
import importlib.util,json,resource,time
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage29_fixed',R/'run-server.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def requests(port,directory,result,api):
 prompts=[]
 for i,name in enumerate(['ALPHA','BETA','GAMMA']):
  prompts.append(name+' report: read the numbered facts.\n'+''.join(f'Entry {j}: the checkpoint value is {1000+i*100+j}; the label is {name.lower()}.\n' for j in range(24))+'\nExplain how to check these entries, then list the first ten checkpoint values.\nAnswer:\n')
 for phase,cache in [('fresh1',False),('reuse1',True),('reuse2',True),('fresh2',False)]:
  for i,prompt in enumerate(prompts):
   payload={'prompt':prompt,'n_predict':64,'temperature':0,'seed':1234,'top_k':40,'top_p':.95,'min_p':.05,'cache_prompt':cache,'return_tokens':True,'n_probs':5,'post_sampling_probs':False}
   base.save(directory/f'{phase}-{i}-request.json',payload);started=time.monotonic()
   response=api(port,'/completion',payload);base.save(directory/f'{phase}-{i}-response.json',response)
   if 'error' in response or not response.get('tokens'):raise RuntimeError(str(response))
   item={'phase':phase,'prompt':i,'slot':response.get('id_slot'),'wall_s':time.monotonic()-started,'timings':response.get('timings'),'tokens':len(response['tokens'])}
   result['requests'].append(item);base.save(directory/'result.json',result);print('RAM_REQUEST',phase,i,json.dumps(item['timings']),flush=True)
base.requests=requests
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 for label,kind in [('dflash-ram','draft-dflash'),('mtp-ram','draft-mtp')]:
  result=base.run(label,kind,1)
  p=R/'server-runs'/label/'control-manifest.json';meta=json.loads(p.read_text());meta['request_schedule']='fully sequential, automatic slots; three long distinct prompts; fresh1/reuse1/reuse2/fresh2';meta['driver_sources'][str(Path(__file__))]=base.sha(Path(__file__));base.save(p,meta)
  if result['status']!='passed':raise SystemExit(1)
