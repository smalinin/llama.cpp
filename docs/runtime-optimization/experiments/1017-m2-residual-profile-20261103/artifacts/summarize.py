from pathlib import Path
import json,statistics,hashlib,re
P=Path(__file__).resolve().parent;r=json.loads((P/'results.json').read_text());a=json.loads((P/'analysis.json').read_text());assert r['complete'];native=r['variants'][0];profile=r['variants'][1]
assert native['variant']=='native'and profile['variant']=='sidecar';out={'complete':True,'runtime_head':r['head'],'native_controls':[],'captures':[],'memory':{}}
for v in r['variants']:
 assert v['complete']and v['exit_code']==0 and not v['warnings']and not v.get('forced_kill')and not v['floor_hit']
 assert 'LLAMA_DSV41_PREFILL_SKIP'not in v['environment']
 assert len(v['skip_lines'])==len(v['seed_lines']) and len(v['skip_lines'])>0
 vals=[x['min_free']for phase in ['load','inference']for x in v['memory'][phase].values()if isinstance(x,dict)and'min_free'in x]
 assert vals and min(vals)>=512*1048576 and not v['memory'].get('error')
 out['memory'][v['variant']]={'all_run_min_free_mib':min(vals)/1048576,'skip_seed_calls':len(v['skip_lines'])}
for n in [512,8192,98304]:
 rows=[q for v in r['variants']for q in v['requests']if q['n']==n];assert len({q['tokens_sha256']for q in rows})==1
 assert len({tuple(q['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])for q in rows})==1
 q=next(q for q in native['requests']if q['n']==n and q['kind']=='native-control');out['native_controls'].append({'n':n,'prompt_per_second':q['response']['timings']['prompt_per_second'],'prompt_ms':q['response']['timings']['prompt_ms'],'output_n':16,'repeats':1})
for x in a['variants'][0]['cases']:
 if 'gpu'not in x:continue
 n=x['n'];g=x['gpu'];assert not g['unmapped'],g['unmapped'];q=next(q for q in profile['requests']if q['n']==n and q['trace']);controls=[q['response']['timings']['prompt_ms']for q in profile['requests']if q['n']==n and q['kind']in['pre-control','post-control']]
 families=[z for z in g['kernel_families']if z['role']=='target_prefill'];s=sum(z['sum_ms']for z in families);byrole={role:sum(z['sum_ms']for z in g['kernel_families']if z['role']==role)for role in sorted({z['role']for z in g['kernel_families']})}
 union=g['gpu_union_ms'].get('target_prefill',0);device=sum(z['ms']for z in g['per_device_gpu_union_ms']if z['role']=='target_prefill')
 row={'n':n,'profile_prompt_ms':x['timings']['prompt_ms'],'profile_prompt_tps':x['timings']['prompt_per_second'],'capture_overhead_pct':100*(x['timings']['prompt_ms']/statistics.median(controls)-1)if controls else None,'control_prompt_tps_range':[n*1000/max(controls),n*1000/min(controls)]if controls else None,'kv_only_ubatches':q['kv_only_ubatches'],'draft_seed_rows':q['draft_seed_rows'],'target_kernel_sum_ms':s,'target_families':[{**z,'pct_kernel_sum':100*z['sum_ms']/s}for z in families],'kernel_sum_by_role_ms':byrole,'target_gpu_activity_union_ms':union,'target_average_active_gpus_during_gpu_activity':device/union if union else None,'per_device_gpu_union_ms':g['per_device_gpu_union_ms'],'cpu':x['cpu'],'copies':g['copies'],'top_transfers':g['transfers'][:25],'sync_api_union_ms':g['sync_api_union_ms'],'top_target_kernels':[z for z in g['kernels']if z['role']=='target_prefill'][:12],'top_target_apis':[z for z in g['api']if z['role']=='target_prefill'][:12]};out['captures'].append(row)
assert len(out['captures'])==3
out['request_count']=sum(len(v['requests'])for v in r['variants']);assert out['request_count']==16
out['outputs_and_draft_counters_exact']=True
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out['captures']:
 print('N',x['n'],'capture overhead',x['capture_overhead_pct'],'families',[(z['family'],round(z['sum_ms'],1),round(z['pct_kernel_sum'],1))for z in x['target_families']],'concurrency',x['target_average_active_gpus_during_gpu_activity'])
print('MEMORY',out['memory'])
