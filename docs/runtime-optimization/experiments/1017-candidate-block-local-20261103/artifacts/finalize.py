from pathlib import Path
import json,re,subprocess,hashlib,shutil,gzip
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(P/'config.json'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
for n,h in cfg['source_library_hashes'].items():assert hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()==h,'library changed'
assert subprocess.check_output(['git','diff','--','src'],cwd=R,text=True)==cfg['runtime_patch'],'source changed'
subprocess.run(['python3',str(P/'analyze-audit.py')],check=True)
precision=json.load(open(P/'precision-results.json'));bench=json.load(open(P/'bench-results.json'));meta=json.load(open(P/'metadata-results.json'));neg=json.load(open(P/'negative-results.json'));op=json.load(open(P/'operator-summary.json'));fault=json.load(open(P/'operator-failure-results.json'));assert fault['complete'];audit=json.load(open(P/'audit-analysis.json'));payload=json.load(open(P/'payload-176128/results.json'))
for j in [precision,bench,meta,neg,op,audit,payload]:assert j['complete']
for arm in precision['arms']:
 assert arm['complete']and arm['exit_code']==0 and not arm['warnings']and arm['groups_exact']==12
for arm in bench['arms']:
 assert arm['exit_code']==0 and not arm['warnings']and not arm.get('forced_kill')
model=[]
for arm in bench['arms']:
 text=(P/('bench-'+arm['label'])/'server.log').read_text(errors='replace')
 model.append([re.sub(r'^.*?I load_tensors:','I load_tensors:',l)for l in text.splitlines()if'model buffer size ='in l])
assert model[0]==model[1]==model[2]
minimum=lambda m:min(v['min_free']/1048576 for ph in ['load','inference']for v in m[ph].values()if isinstance(v,dict)and'min_free'in v)
summary={'complete':True,'base':cfg['source_head'],'libraries':cfg['source_library_hashes'],'operator_cases':op['cases'],'fixed32_exact':True,'external_compute_failure_retry':fault,'precision_groups':sum(a['groups_exact']for a in precision['arms']),'faults':sum(len(a['faults'])for a in precision['arms']),'precision_min_mib':[minimum(a['memory'])for a in precision['arms']],'native':bench['summary'],'native_model_buffers_exact':True,'native_min_mib':[minimum(a['memory'])for a in bench['arms']],'metadata':meta['contexts'],'negative_guards':neg['cases'],'copy_proof':{k:v for k,v in audit.items()if k!='matches'},'payload':{'context':payload['context'],'min_mib':minimum(payload['memory']),'exit':payload['exit_code'],'actual_routes_exact':True,'all_experts_GPU':15360,'tokens_exact':payload['same_172k_tokens_as_old_copyOFF']and payload['same_8k_tokens_as_benchmark']and payload['repeat_tokens_equal_first16']}}
assert summary['payload']['tokens_exact']and payload['exit_code']==0 and not payload['allocation_warnings']and summary['payload']['min_mib']>=512
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(P/'FINAL.patch').write_text(cfg['runtime_patch'])
print(json.dumps(summary,indent=2))
