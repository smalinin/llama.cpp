#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
R=Path(__file__).resolve().parent;D=R/'free-runs'
CONFIGS=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n1-remaining','candidate-n3']
SHORT=['baseline','svg','python','explain'];LONG=['dense','ratio1','ratio2']
LABELS=[f'{p}-greedy-{r}' for r in [1,2] for p in SHORT]+[f'{p}-{state}' for p in LONG for state in ['fresh','cache']]
def read(p):return json.loads(p.read_text())
def get(config,label):return read(D/config/f'{label}-response.json')
def comparison(a,b):
 x,y=a['tokens'],b['tokens'];first=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)) if len(x)!=len(y) else None)
 return {'tokens_identical':x==y,'content_identical':a['content']==b['content'],'stop_metadata_identical':all(a.get(k)==b.get(k) for k in ['stop_type','stopping_word','truncated']),'reference_tokens':len(x),'actual_tokens':len(y),'first_difference':first,'reference_window':None if first is None else x[max(0,first-2):first+3],'actual_window':None if first is None else y[max(0,first-2):first+3]}
def equal(c):return all(c[k] for k in ['tokens_identical','content_identical','stop_metadata_identical'])
def events(text):return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',m.group(1))} for m in re.finditer(r'DS14_EVENT (.*)',text)]
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
out={'scope':'Free greedy server correctness with the unchanged Stage19/20 callback. Sparse counters describe the effective scalar query predicate, not a new kernel profiler measurement. Timing includes callback overhead.','against_native':{},'probability_output_controls':{},'native_prior_controls':{},'fresh_vs_cache':{},'cache_reuse':{},'acceptance':{},'events':{},'requests':{},'request_errors':{},'extent_failures':{}}
for config in CONFIGS:
 if not (D/config/'result.json').exists() or read(D/config/'result.json')['status']=='running':
  if args.partial:continue
  raise RuntimeError('incomplete '+config)
 result=read(D/config/'result.json');assert result['server_exit_code']==0
 expected={'snapshot-off':14,'integration-off':14,'candidate-off':14,'candidate-n1':9,'candidate-n1-remaining':2,'candidate-n3':11}
 assert len(result['requests'])==expected[config],(config,result)
 if result['status']=='failed':
  failure=read(D/config/'request-failure.json');assert failure['http_status']==500 and 'unsupported cache extent' in failure.get('body',failure.get('server_message',''))
  out['request_errors'][config]=failure
  lines=[l for l in (D/config/'server.log').read_text(errors='replace').splitlines() if l.startswith('DS21_EXTENT ')]
  if lines:
   assert len(lines)==1
   out['extent_failures'][config]={'line':lines[0],**{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',lines[0])}}
 log=(D/config/'server.log').read_bytes();all_events=events(log.decode(errors='replace'))
 for item in result['requests']:
  label=item['request'];response=get(config,label);native=get('snapshot-off',label)
  request=read(D/config/f'{label}-request.json');ref_request=read(D/'snapshot-off'/f'{label}-request.json');assert request==ref_request
  assert item['tokens_sha256']==hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
  out['against_native'][f'{config}/{label}']=comparison(native,response)
  t=response['timings'];out['acceptance'][f'{config}/{label}']={'draft_n':t.get('draft_n',0),'draft_n_accepted':t.get('draft_n_accepted',0)}
  if item['cache_prompt']:
   assert t['cache_n']>0,(config,label,t)
   out['cache_reuse'][f'{config}/{label}']={'cache_n':t['cache_n'],'prompt_n':t['prompt_n'],'prompt_tokens':item['prompt_tokens']}
  es=events(log[item['log_start_offset']:item['log_end_offset']].decode(errors='replace'))
  gen=[e for e in es if e['generation']];pre=[e for e in es if not e['generation']]
  if not config.startswith('snapshot-'):
   assert gen and all(e['ret']==0 for e in es)
   assert all(e['precision']==e['attention']==e['upgate']==e['matmul']==e['padding_crops']==e['compressed_crops']==0 for e in pre)
   if config=='integration-off':assert all(not e['enabled'] and e['precision']==e['attention']==e['upgate']==e['matmul']==0 for e in es)
   else:
    assert all(e['enabled'] for e in es)
    for e in gen:
     expected=40*e['width'] if e['width']>1 else 0
     assert e['precision']>0 and e['attention']==e['upgate']==expected and e['matmul']==4*expected+(7*e['width'] if e['width']>1 else 0),e
     expected_ratio1=20*e['width']
     expected_ratio2=18*e['width']
     assert e['r1_dense']+e['r1_sparse']==expected_ratio1 and e['r2_dense']+e['r2_sparse']==expected_ratio2,e
   summary={'prefill_calls':len(pre),'generation_calls':len(gen),'generation_widths':dict(Counter(str(e['width']) for e in gen)),'decode_min':min(e['pos'] for e in gen),'decode_max':max(e['pos']+e['width']-1 for e in gen),'prefill_interventions_zero':True,'predicate_counters_only_width_gt_1':True}
   summary.update({k:sum(e[k] for e in gen if not k.startswith('r') or e['width']>1) for k in ['attention','upgate','matmul','padding_crops','compressed_crops','r1_dense','r1_sparse','r2_dense','r2_sparse']})
   out['requests'][f'{config}/{label}']=summary
 out['events'][config]={'total_calls':len(all_events),'all_ret_zero':all(e['ret']==0 for e in all_events)}
 for p in SHORT:
  if not (D/config/f'{p}-greedy-1-response.json').exists():continue
  out['probability_output_controls'][f'{config}/{p}']=comparison(get(config,f'{p}-greedy-1'),get(config,f'{p}-greedy-2'))
 for p in LONG:
  if not (D/config/f'{p}-cache-response.json').exists():continue
  out['fresh_vs_cache'][f'{config}/{p}']=comparison(get(config,f'{p}-fresh'),get(config,f'{p}-cache'))
for p in SHORT:
 out['native_prior_controls'][p]=comparison(read(R.parent/'stage18/free-runs/snapshot-off'/f'{p}-greedy-1-response.json'),get('snapshot-off',f'{p}-greedy-1'))
checks={k:all(equal(c) for c in out[k].values()) for k in ['against_native','probability_output_controls','native_prior_controls','fresh_vs_cache']}
out['checks']=checks;out['overall_server_compatibility_passed']=not out['request_errors'] and all(checks.values());out['completed_configs']=list(out['events']);out['comparisons']=len(out['against_native'])
(R/('free-partial-summary.json' if args.partial else 'free-summary.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'overall_server_compatibility_passed':out['overall_server_compatibility_passed'],'request_errors':out['request_errors'],'extent_failures':out['extent_failures'],'checks':checks,'configs':out['completed_configs'],'comparisons':out['comparisons'],'cache_reuse':out['cache_reuse'],'differences':{k:v for k,v in out['against_native'].items() if not equal(v)}},indent=2))
