from pathlib import Path
import json,hashlib
P=Path(__file__).parent;cfg=json.load(open(P/'config.json'));a=json.load(open(P/'payload-admission.json'));current={};rows=[]
for ctx,flag in [(100352,'0'),(100352,'1'),(176128,'1'),(163840,'1'),(159744,'1')]:
 f=P/f'metadata-{ctx}-{flag}.json';j=json.load(open(f));assert j['complete'];current[(ctx,flag)]=j
 assert j['gpu_complete_bundles']==15360 and j['routed_host_bytes']==0
 sourceobs = 0 if flag == '0' else 1
 for i,m in enumerate(j['memory']):
  if ctx==100352:expected=a['candidates'][0]['rows'][i]['observations'][sourceobs]['projected_free']
  else:expected=next(x for x in a['candidates']if x['context']==ctx)['rows'][i]['projected_free']
  assert m['projected_free']==expected,(ctx,flag,i,m['projected_free'],expected)
 trace=[json.loads(s)for s in(P/f'meta-trace-{ctx}-{flag}.jsonl').read_text().splitlines()];ps={(r['name'],r['op'],r['backend'],tuple(r['ne']))for r in trace};assert any(x[0].startswith('candidate_local_repeat-')for x in ps)==(flag=='1');assert {x[2]for x in ps if x[0]=='candidate_block_score-20'and x[3][1]>6}=={'CUDA3'}
 rows.append({'context':ctx,'suffix':flag,'metadata_complete':True,'source':str(f),'current_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'original_admission_hash':a['reference_hashes'].get(str(f)),'all_per_device_projected_free_exact_to_original_admission':True,'pool_env':None,'block_env':None if flag=='1'else'0','trace_comparison':'sets of placements/shapes; original trace prefix plus replay append retained, no execution counts inferred'})
a0=current[(100352,'0')];a1=current[(100352,'1')];assert a0['routes']==a1['routes']
for key,x in current.items():assert x['routes']==a0['routes']
for x,y in zip(a0['memory'],a1['memory']):
 for k in ['main_model','main_context','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k]
payload=json.load(open(P/'payload-163840/results.json'));assert payload['actual_placement']['layers']==a1['routes']
raw=json.load(open(P/'metadata-runs.json'));out={'complete':True,'recovery':True,'reason':'Post-GPU summary import unexpectedly executed local queue.py instead of stdlib queue. Metadata was replayed and the original runner manifest overwritten/incomplete. Original admission hashes retained; metadata elapsed timings changed. queue.py renamed run-gates.py. Native/precision/payload results remained complete and were not rerun.','raw_manifest_complete':raw['complete'],'raw_manifest_runs':len(raw.get('runs',[])),'cases':rows,'per_tensor_routes_exact':True,'current_default_actual_placement_exact':True,'runtime_or_models_changed':False};(P/'metadata-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
