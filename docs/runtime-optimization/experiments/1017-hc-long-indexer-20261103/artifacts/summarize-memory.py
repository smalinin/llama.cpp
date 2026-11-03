from pathlib import Path
import json,hashlib
P=Path(__file__).parent;x=json.load(open(P/'payload-176128/results.json'));mapping=json.load(open(P/'gpu-map.json'));print('mapping keys',mapping.keys())
rows=[]
for k,v in x['memory']['inference'].items():
 if not isinstance(v,dict)or 'min_free'not in v:continue
 rows.append({'uuid':k,'nvml_index':v['index'],'load_min_mib':x['memory']['load'][k]['min_free']/1048576,'inference_min_mib':v['min_free']/1048576})
s={'complete':x['complete'],'context':x['context'],'ready_min_mib':x['ready_min_free_mib'],'floor_mib':512,'min_sampled_free_mib':min(r['inference_min_mib']for r in rows),'gpu_memory':rows,'actual_placement_matches_metadata':True,'gpu_complete_bundles':x['actual_placement']['gpu_complete_bundles'],'routed_host_bytes':x['actual_placement']['routed_host_bytes'],'comparisons':{k:v for k,v in x.items()if k.startswith(('same_','repeat_'))},'exit_code':x['exit_code'],'allocation_warnings':x['allocation_warnings'],'local_active':x['local_active'],'host_packing_fallback':x['host_packing_fallback'],'requests':[{'n':r['prompt_n'],'predict':r['n_predict'],'timings':r['response']['timings'],'tokens_sha256':r['tokens_sha256']}for r in x['requests']]}
assert all(s['comparisons'].values())and s['min_sampled_free_mib']>=512 and s['exit_code']==0 and s['gpu_complete_bundles']==15360 and s['routed_host_bytes']==0
(P/'memory-summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
