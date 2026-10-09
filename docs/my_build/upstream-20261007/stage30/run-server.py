#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,importlib.util,json,resource
R=Path(__file__).resolve().parent;S29=R.parent/'stage29'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ram=load('stage29_ram',S29/'run-ram.py')
def requests(port,directory,result,api):
 ram.requests(port,directory,result,api)
 if EXTRA_RESIDENT:
  for i in range(3):
   for phase,cache in [('resident-fresh1',False),('resident-reuse1',True),('resident-reuse2',True),('resident-fresh2',False)]:
    payload=json.loads((directory/f'fresh1-{i}-request.json').read_text());payload.update(id_slot=2-i,cache_prompt=cache)
    ram.base.save(directory/f'{phase}-{i}-request.json',payload)
    response=api(port,'/completion',payload);ram.base.save(directory/f'{phase}-{i}-response.json',response)
    assert 'error' not in response and response.get('tokens')
    result['requests'].append({'phase':phase,'prompt':i,'slot':response.get('id_slot'),'timings':response.get('timings'),'tokens':len(response['tokens'])})
    ram.base.save(directory/'result.json',result);print(phase,i,response['timings'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--build',required=True);p.add_argument('--kind',default='none');p.add_argument('--cache',type=int,default=0);p.add_argument('--resident',action='store_true');a=p.parse_args();EXTRA_RESIDENT=a.resident
 b=json.loads((R/(a.build+'-build-manifest.json')).read_text());b['sources']=b['base_sources']
 code=(S29/'run-server.py').read_text();code=code.replace("b=json.loads((PREV/'final-build-manifest.json').read_text());hashes=b['candidate_hashes']", "b=BUILD;hashes=b['candidate_hashes']")
 ns={'__name__':'stage30_runner','__file__':str(Path(__file__)),'BUILD':b};exec(compile(code,str(S29/'run-server.py'),'exec'),ns);ns.update(R=R,LIB=R/(a.build+'-bin'),requests=requests)
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));result=ns['run'](a.label,a.kind,a.cache)
 meta_path=R/'server-runs'/a.label/'control-manifest.json';meta=json.loads(meta_path.read_text());meta.update(build_manifest=b,extra_resident=EXTRA_RESIDENT,request_schedule='sequential Stage29 RAM replay; optional same-prompt resident repetitions');meta['driver_sources'].update({str(p):ram.base.sha(p) for p in [S29/'run-server.py',S29/'run-ram.py']});ram.base.save(meta_path,meta)
 if result['status']!='passed':raise SystemExit(1)
