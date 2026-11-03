from pathlib import Path
import json,re,hashlib,statistics,subprocess,collections
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');cfg=json.load(open(P/'config.json'));b=json.load(open(P/'bench-results.json'));long=json.load(open(P/'long-summary.json'));fresh=json.load(open(P/'fresh-on-summary.json'));guard=json.load(open(P/'final-guard-summary.json'));op=json.load(open(P/'operator-final-results.json'));payload=json.load(open(P/'payload-176128/results.json'));audit=json.load(open(P/'audit-analysis.json'));meta=json.load(open(P/'metadata-results.json'))
assert all(x['complete']for x in[b,long,fresh,guard,op,payload,audit,meta]);assert len(op['cases'])==80 and op['repeat32']['exact']and op['actual_placement_checked'];assert b['library_hashes']==cfg['source_library_hashes']==payload['binary_sha256']
for n,h in cfg['source_library_hashes'].items():assert hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()==h
assert (R/'src/llama-context.cpp').read_bytes()==(P/'llama-context.final.cpp').read_bytes()
def mem_min(x):return min(v['min_free']/1048576 for ph in ['load','inference']for v in x[ph].values()if isinstance(v,dict)and'min_free'in v)
weights=[];memory=[]
for a in b['arms']:
 assert a['complete']and a['exit_code']==0 and not a['warnings']and not a.get('forced_kill')and not a['floor_hit']
 t=(P/f"bench-{a['label']}/server.log").read_text(errors='replace');weights.append([l.split(' I ',1)[-1]for l in t.splitlines()if'model buffer size ='in l]);memory.append({'arm':a['label'],'min_free_mib':mem_min(a['memory'])});assert memory[-1]['min_free_mib']>=512
assert weights[0]==weights[1]==weights[2],'model buffers differ'
assert payload['exit_code']==0 and not payload['allocation_warnings']and not payload.get('floor_breach')and not payload.get('forced_kill')and payload['local_active']and not payload['host_packing_fallback'];assert payload['same_8k_tokens_as_benchmark']and payload['repeat_tokens_equal_first16']and payload['same_172k_tokens_as_old_copyOFF'];assert mem_min(payload['memory'])>=512
log=(P/'operator-final.log').read_text(errors='replace');fixed=[]
for mode in [0,1]:
 ss=log.split(f'FIXED_BEGIN {mode}\n',1)[1].split(f'FIXED_END {mode}\n',1)[0];rr=[]
 for line in ss.splitlines():
  if line.startswith('cuda_graph_stats:'):
   x={k:int(v)for k,v in re.findall(r'(\w+)=(\d+)',line)};rr.append(x)
 assert len(rr)==2
 needed=[r for r in rr if mode==1 or r['device']==1]
 assert all(r['compute_calls']>=32 and r['compute_calls']%32==0 and r['update_failures']==0 and r['captures']==r['compute_calls']//32 and r['launches']==31*r['captures'] and r['eager_properties_changed']==0 for r in needed),rr
 fixed.append({'mode':mode,'stats':rr})
summary={'complete':True,'head':cfg['source_head'],'flag':'LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1','default':'OFF','source_sha256':{str(p.relative_to(R)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()for p in[R/'src/llama-context.cpp',R/'src/models/deepseek41.cpp']},'libraries':cfg['source_library_hashes'],'quality':{'mutable_groups':long['result_groups'],'fresh_on_groups':fresh['groups'],'rollback_workflows':long['rollback_workflows'],'nll':long['nll'],'mutable_retained_capacity_warning':long['mutable_fixture_warning'],'fresh_on_warnings':fresh['warnings'],'fresh_on_min_free_mib':fresh['min_free_mib'],'raw_fresh_postprocess_error':fresh['raw_postprocess_error']},'final_guard':guard,'operator':{'cases':80,'fixed32':fixed,'initial_run_accepted':False,'initial_reason':'Implicit allocation reset manual hints; both routes ran onCUDA0. Final explicit allocation verifies actual backend IDs.'},'native':{'requests':sum(len(a['requests'])for a in b['arms']),'model_buffers_equal':True,'memory':memory,'summary':b['summary']},'actual176128':{'min_free_mib':mem_min(payload['memory']),'gpu_complete_bundles':payload['actual_placement']['gpu_complete_bundles'],'routed_host_bytes':payload['actual_placement']['routed_host_bytes'],'exit_code':payload['exit_code'],'warnings':payload['allocation_warnings'],'tokens_match':True,'placement_matches_metadata':True},'copies':audit['wide_summary'],'fallback_copies':audit['fallback_summary'],'metadata':meta}
assert summary['actual176128']['gpu_complete_bundles']==15360 and summary['actual176128']['routed_host_bytes']==0
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in ['metadata','libraries','source_sha256','final_guard']},indent=2))
