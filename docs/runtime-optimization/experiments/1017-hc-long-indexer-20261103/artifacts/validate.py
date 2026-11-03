from pathlib import Path
import json,hashlib,subprocess
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');m=json.load(open(P/'memory-summary.json'));r=json.load(open(P/'profile-results.json'));mtd=json.load(open(P/'metadata-results.json'));old=json.load(open('/tmp/dsv41-m2-residual-profile-20261103/results.json'));v=r['variants'][0];es=[json.loads(l)for l in(P/'sidecar/events.jsonl').open()];checks={}
assert m['complete']and r['complete']and v['complete']and v['exit_code']==0 and not v['warnings']and not v.get('forced_kill')and not v['floor_hit'];checks['clean_exit']=True
for n,h in r['binary_sha256'].items():assert hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()==h
checks['binary_unchanged']=True
comp=[]
for n in [8192,98304]:
 ref=next(q for a in old['variants']if a['variant']=='native'for q in a['requests']if q['n']==n and q['kind']=='native-control');qs=[q for q in v['requests']if q['n']==n]
 assert len(qs)==4
 for q in qs:
  assert q['response']['tokens']==ref['response']['tokens'];assert all(q['response']['timings'][k]==ref['response']['timings'][k]for k in ['draft_n','draft_n_accepted']);assert q['response']['timings']['cache_n']==0
 comp.append({'prefix':n,'requests':len(qs),'native_reference_ids_and_draft_counters_exact':True,'pre_ms':qs[1]['response']['timings']['prompt_ms'],'profile_ms':qs[2]['response']['timings']['prompt_ms'],'post_ms':qs[3]['response']['timings']['prompt_ms'],'target_ubatches':qs[2]['target_ubatches'],'kv_only_ubatches':qs[2]['kv_only_ubatches'],'draft_seed_rows':qs[2]['draft_seed_rows']})
checks['native_reference_comparisons']=comp
actual=next(json.loads(l)for l in(P/'sidecar/placement.jsonl').open()if json.loads(l)['arch']=='deepseek41');md=json.load(open(P/'metadata-100352-default.json'));assert actual['layers']==md['routes'];assert actual['gpu_complete_bundles']==15360 and actual['routed_host_bytes']==0
checks['actual_profile_placement_exact_to_metadata']=True;checks['candidate_topology']=actual['candidate'];checks['topology']=actual['topology']
checks['profile_min_free_mib']=min(x['min_free']/1048576 for phase in v['memory'].values()if isinstance(phase,dict)for x in phase.values()if isinstance(x,dict)and'min_free'in x);assert checks['profile_min_free_mib']>=512
checks['default_explicit1_no_alloc_equal']=mtd['default_explicit1_budget_exact'];assert checks['default_explicit1_no_alloc_equal']
checks['metadata_copy_wrappers_observed']=sum(e['event']=='ix_copy_metadata'for e in es);checks['copy_wrapper_scope_note']='Scheduler uses direct backend iface.cpy_tensor_async for these cross-GPU inputs; generic ggml_backend_tensor_copy_async wrapper does not observe that path. Physical attribution uses separate metadata-constrained same-thread next-backend/API correlation; it does not claim intercepted CUDA pointer proof.'
checks['complete']=True;(P/'validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({k:v for k,v in checks.items()if k!='topology'},indent=2))
