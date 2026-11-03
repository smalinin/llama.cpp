from pathlib import Path
import json,hashlib,re
P=Path(__file__).parent;r=P/'payload-176128';j=json.load(open(r/'results.json'))
assert len(j['requests'])==3 and j['exit_code']==0
assert j.get('error')in[None,"KeyError('prompt_n')"],j.get('error')
cfg=json.load(open(P/'config.json'));assert j['binary_sha256']==cfg['source_library_hashes'];meta=json.load(open(P/'metadata-176128-default.json'));assert j['actual_placement']['layers']==meta['routes'];assert j['actual_placement']['gpu_complete_bundles']==15360 and j['actual_placement']['routed_host_bytes']==0
assert not j.get('floor_breach')and not j.get('forced_kill')and not j['allocation_warnings']
assert j['local_active']and not j['host_packing_fallback']and not j['memory'].get('error')
assert all(v['min_free']>=512*1048576 for phase in ['load','inference']for v in j['memory'][phase].values()if isinstance(v,dict)and'min_free'in v)
a=json.load(open('/tmp/dsv41-dspark-tail-screen-20261103/bench-results.json'))
b=next(x for x in a['arms']['on']['requests']if x['n']==8192 and not x['warmup'])
old=json.load(open('/tmp/dsv41-feature-local-long-20261103/payload-176128/results.json'))
assert json.load(open(r/'inputs.json'))['tokens']==json.load(open('/tmp/dsv41-feature-local-long-20261103/payload-176128/inputs.json'))['tokens']
out={'complete':True,'raw_results':'payload-176128/results.json','harness_issue':'Post-request comparison used prompt_n instead of n in historical benchmark schema. All3 requests and server cleanup finished first. Raw error/result preserved; runner fixed on disk for reproduction. No GPU rerun needed for this comparison.','comparisons':{'8k256_vs_prior_m2':j['requests'][0]['response']['tokens']==b['response']['tokens'],'8k_audit_vs_first16':j['requests'][2]['response']['tokens']==j['requests'][0]['response']['tokens'][:16],'172k16_vs_prior_pre_m2_local':j['requests'][1]['response']['tokens']==old['requests'][1]['response']['tokens']},'memory':j['memory'],'sampled_min_free_mib':min(v['min_free']for phase in ['load','inference']for v in j['memory'][phase].values()if isinstance(v,dict)and'min_free'in v)/1048576,'exit_code':j['exit_code'],'requests':[{'prompt_n':q['prompt_n'],'predict':q['n_predict'],'timings':q['response']['timings'],'tokens_sha256':q['tokens_sha256']}for q in j['requests']]}
assert all(out['comparisons'].values()),out['comparisons']
log=(r/'server.log').read_text(errors='replace');out['m2_lines']=[l for l in log.splitlines()if re.search('KV-only|kv-only|prefill_skip|dspark_prefill_tail',l,re.I)]
(P/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print('VALIDATED',out['comparisons'],out['sampled_min_free_mib'])
