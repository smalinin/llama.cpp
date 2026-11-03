from pathlib import Path
import json,collections
P=Path(__file__).parent;out={'complete':False,'cases':[]}
for ctx in [100352,176128]:
 off=json.load(open(P/f'metadata-{ctx}-0.json'));on=json.load(open(P/f'metadata-{ctx}-1.json'));assert off['complete']and on['complete']
 for k in ['routes','routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles']:assert off[k]==on[k],k
 deltas=[]
 for a,b in zip(off['memory'],on['memory']):
  for k in ['main_model','main_context','draft_and_taps_model','draft_and_taps_context']:assert a[k]==b[k],k
  deltas.append({'device':a['device'],'combined_compute_delta_mib':((b['main_compute']+b['draft_and_taps_compute'])-(a['main_compute']+a['draft_and_taps_compute']))/1048576})
 counts=[]
 for flag in ['0','1']:
  rows=[json.loads(l)for l in(P/f'meta-trace-{ctx}-{flag}.jsonl').open()];wide=[r for r in rows if r['ne'][1]==512];by=collections.defaultdict(set)
  for r in wide:by[r['name']].add(r['backend'])
  assert by['idx_score-20']=={'CUDA2'};assert by['candidate_score_cont-20']==by['candidate_block_score-20']==({'CUDA2'}if flag=='1'else{'CUDA3'});assert by['candidate_pinned_block_score-20']==by['candidate_mask-20']=={'CUDA3'}
  counts.append({'flag':flag,'wide_backends':{k:sorted(v)for k,v in by.items()},'small_ne1':sorted(set(r['ne'][1]for r in rows if r['ne'][1]<7))})
 out['cases'].append({'context':ctx,'combined_compute_delta':deltas,'trace':counts})
out['complete']=True;(P/'metadata-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
