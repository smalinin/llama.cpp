#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent;D=ROOT/'free-runs'
configs=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3'];prompts=['baseline','svg','python','explain']
parser=argparse.ArgumentParser();parser.add_argument('--controls-only',action='store_true');args=parser.parse_args()
if args.controls_only:configs=configs[:3]
def get(config,prompt,repeat):return json.loads((D/config/f'{prompt}-greedy-{repeat}-response.json').read_text())
def comparison(a,b):
 x,y=a['tokens'],b['tokens'];first=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)) if len(x)!=len(y) else None)
 detail={}
 if first is not None:
  for label,data in [('reference',a),('actual',b)]:
   probabilities=data.get('completion_probabilities',[])
   if first<len(probabilities):detail[label+'_first_difference_probabilities']=probabilities[first]
 return {**detail,'tokens_identical':x==y,'content_identical':a['content']==b['content'],'stop_metadata_identical':all(a.get(k)==b.get(k) for k in ['stop_type','stopping_word','truncated']),'reference_tokens':len(x),'actual_tokens':len(y),'first_difference':first,'reference_stop_type':a.get('stop_type'),'actual_stop_type':b.get('stop_type'),'reference_window':None if first is None else x[max(0,first-2):first+3],'actual_window':None if first is None else y[max(0,first-2):first+3]}
out={'scope':'Free greedy token-ID/content equality to current native; n_probs5 and n_probs0 compared. Callback overhead is included in timings.','native_prior_controls':{},'probability_output_controls':{},'against_native':{},'callback_events':{},'acceptance':{}}
for config in configs:
 result=json.loads((D/config/'result.json').read_text());assert result['status']=='passed' and result['server_exit_code']==0
 for prompt in (['baseline'] if config=='snapshot-n3' else prompts):
  a,b=get(config,prompt,1),get(config,prompt,2)
  out['probability_output_controls'][f'{config}-{prompt}']=comparison(a,b)
  for repeat in [1,2]:
   data=get(config,prompt,repeat);native=get('snapshot-off',prompt,repeat)
   out['against_native'][f'{config}-{prompt}-{repeat}']=comparison(native,data)
   t=data['timings'];out['acceptance'][f'{config}-{prompt}-{repeat}']={'draft_n':t.get('draft_n',0),'draft_n_accepted':t.get('draft_n_accepted',0),'acceptance':t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n',0) else None,'predicted_per_second_with_diagnostic_overhead':t['predicted_per_second']}
 if not config.startswith('snapshot-'):
  events=[]
  for match in re.finditer(r'DS14_EVENT (.*)',(D/config/'server.log').read_text(errors='replace')):
   event={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',match.group(1))};events.append(event)
  assert events and all(e['ret']==0 for e in events)
  prefill=[e for e in events if not e['generation']];gen=[e for e in events if e['generation']]
  assert prefill and gen
  assert all(e['attention']==e['upgate']==e['matmul']==e['precision']==0 for e in prefill)
  if config=='integration-off':assert all(not e['enabled'] and e['attention']==e['upgate']==e['matmul']==e['precision']==0 for e in events)
  else:
   assert all(e['enabled'] for e in events)
   assert all(e['precision']>0 for e in gen)
   for e in gen:
    expected=40*e['width'] if e['width']>1 else 0
    assert e['attention']==e['upgate']==expected and e['matmul']==4*expected+(7*e['width'] if e['width']>1 else 0),e
  out['callback_events'][config]={'prefill_calls':len(prefill),'generation_calls':len(gen),'generation_width_histogram':{str(w):sum(e['width']==w for e in gen) for w in sorted({e['width'] for e in gen})},'prefill_all_interventions_zero':True,'attention_queries':sum(e['attention'] for e in gen),'upgate_token_layers':sum(e['upgate'] for e in gen),'matmul_token_results':sum(e['matmul'] for e in gen)}
for prompt in prompts:
 old=json.loads((ROOT.parent/'stage9/n1-server/off'/f'{prompt}-measure-1-response.json').read_text())
 out['native_prior_controls'][prompt]=comparison(old,get('snapshot-off',prompt,2))
assert all(out['against_native'][f'integration-off-{p}-{r}']['tokens_identical'] and out['against_native'][f'integration-off-{p}-{r}']['content_identical'] and out['against_native'][f'integration-off-{p}-{r}']['stop_metadata_identical'] for p in prompts for r in [1,2])
assert all(out['against_native'][f'candidate-off-{p}-{r}']['tokens_identical'] and out['against_native'][f'candidate-off-{p}-{r}']['content_identical'] and out['against_native'][f'candidate-off-{p}-{r}']['stop_metadata_identical'] for p in prompts for r in [1,2])
path=ROOT/('free-control-summary.json' if args.controls_only else 'free-summary.json');path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'prior_native':out['native_prior_controls'],'probability_output_controls':{k:v for k,v in out['probability_output_controls'].items() if not v['tokens_identical']},'candidate_comparisons':{k:v for k,v in out['against_native'].items() if k.startswith('candidate')},'events':out['callback_events']},indent=2))
