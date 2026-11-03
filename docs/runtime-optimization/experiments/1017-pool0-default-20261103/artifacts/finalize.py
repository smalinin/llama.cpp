from pathlib import Path
import json,hashlib,subprocess,re
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(P/'config.json'));q=json.load(open(P/'precision-results.json'));b=json.load(open(P/'bench-results.json'));m=json.load(open(P/'metadata-validation.json'));a=json.load(open(P/'payload-admission.json'));d=json.load(open(P/f"payload-{a['chosen_context']}/results.json"));assert all(x['complete']for x in [q,b,m,a,d])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
for name,h in cfg['source_library_hashes'].items():assert hashlib.sha256((R/'build-aliceai/bin'/name).read_bytes()).hexdigest()==h,name
for name,h in cfg['source_files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
minimum=lambda memory:min(x['min_free']/1048576 for phase in ['load','inference']for x in memory[phase].values()if isinstance(x,dict)and'min_free'in x)
for arm in q['arms']:
 assert arm['exit_code']==0 and not arm['warnings']and arm['groups_exact']==12 and minimum(arm['memory'])>=512
 assert'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in arm['environment']
 if arm['flag']=='default':assert'LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL'not in arm['environment']
 t=json.load(open(P/f"precision-{arm['flag']}-trace.json"));assert t['injected_failures']==2
 raw_log=(P/f"precision-{arm['flag']}.log").read_bytes();assert raw_log.count(b'INJECTED_COMPUTE_FAILURE')==2
buffers=[]
for arm in b['arms']:
 assert arm['exit_code']==0 and not arm['warnings']and not arm['floor_hit']and minimum(arm['memory'])>=512
 assert'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in arm['environment']
 if arm['label']=='on':assert'LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL'not in arm['environment']
 text=(P/('bench-'+arm['label'])/'server.log').read_text(errors='replace');buffers.append([re.sub(r'^.*?I load_tensors:','I load_tensors:',l)for l in text.splitlines()if'model buffer size ='in l])
assert buffers[0]==buffers[1]==buffers[2]
assert d['exit_code']==0 and not d['allocation_warnings']and not d.get('forced_kill')and not d.get('floor_breach')and minimum(d['memory'])>=512
assert d['actual_placement']['gpu_complete_bundles']==15360 and d['actual_placement']['routed_host_bytes']==0
assert d['same_8k_tokens_as_benchmark']and d['repeat_tokens_equal_first16'];assert d['same_172k_tokens_as_old_copyOFF']is not False
historical=json.load(open('/tmp/dsv41-candidate-block-local-20261103/bench-results.json'));refs={x['n']:x for x in historical['arms'][1]['requests']}
for arm in b['arms']:
 for x in arm['requests']:
  ref=refs[x['n']];assert ref['tokens_sha256']==x['tokens_sha256']
  for key in ['draft_n','draft_n_accepted']:assert ref['response']['timings'][key]==x['response']['timings'][key]
summary={'complete':True,'head':cfg['source_head'],'runtime_changed':False,'libraries':cfg['source_library_hashes'],'source_files':cfg['source_files'],'precision':{'groups':24,'corpora':2,'seed_failures':4,'exact_to_prior_reference':True,'nll_delta':0,'activation':[x['activation']for x in q['arms']],'pooling_placement':[x['pooling_placement']for x in q['arms']],'minimum_free_mib':[minimum(x['memory'])for x in q['arms']]},'native':b['summary'],'native_requests':27,'native_exact_to_prior_pool1':True,'model_buffers_exact':True,'summary_recovery':m,'native_minimum_free_mib':[minimum(x['memory'])for x in b['arms']],'memory':{'context':d['context'],'prefix':d['requests'][1]['prompt_n'],'minimum_free_mib':minimum(d['memory']),'actual_GPU_bundles':15360,'routes_exact':True,'exit':0,'same_8K_and_repeat_ids':True,'same_near_to_historical':d['same_172k_tokens_as_old_copyOFF'],'candidate_admission':[{'context':x['context'],'admitted':x['admitted'],'guarded_min_mib':min(r['guarded_estimate_free']for r in x['rows'])/1048576}for x in a['candidates']]},'limits':'Frozen Q3/MXFP8 Engram/Q8KV/BF16 DSparkK3/full/pmin.5/B8192/U512/P1/TopK2048/allGPU placement. Candidate flags unset(defaultONblock/OFFpool). Native98Ksingleton and one arm order; no Maya comparison or new kernel profiling. Lower-ctx near-prefix has no historical token oracle if same_near is null.'}
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
