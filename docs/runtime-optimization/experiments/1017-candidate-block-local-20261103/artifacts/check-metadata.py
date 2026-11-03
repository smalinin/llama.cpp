from pathlib import Path
import json
P=Path(__file__).parent;j={'complete':False,'contexts':[]}
for n in [100352,176128]:
 a,b=[json.load(open(P/f'metadata-{n}-{f}.json'))for f in [0,1]];assert a['complete']and b['complete']and a['routes']==b['routes']
 for k in ['routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles']:assert a[k]==b[k]
 assert a['routed_host_bytes']==0 and a['gpu_complete_bundles']==15360
 delta=[]
 for x,y in zip(a['memory'],b['memory']):
  for k in ['main_model','main_context','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k],(n,k)
  delta.append({'device':x['device'],'compute_delta_mib':(y['main_compute']-x['main_compute'])/1048576})
 rows=[json.loads(l)for l in(P/f'meta-trace-{n}-1.jsonl').read_text().splitlines()];assert not any(r['name']=='candidate_mask-20'and r['ne'][1]>6 for r in rows)
 pairs=sorted({(r['name'],r['backend'])for r in rows if 'candidate_local_repeat' in r['name']});assert pairs==[('candidate_local_repeat-28','CUDA3'),('candidate_local_repeat-36','CUDA4')]
 j['contexts'].append({'context':n,'delta':delta,'sum_compute_delta_mib':sum(x['compute_delta_mib']for x in delta),'repeat_owners':pairs,'legacy_wide_nodes':0,'model_KV_placement_exact':True})
j['complete']=True;(P/'metadata-results.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
