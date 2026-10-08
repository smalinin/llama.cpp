#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',ROOT/'analyze-chain.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
s=json.loads((ROOT/'chain-summary.json').read_text());out={'first_differences':{},'routing':{},'early_boundaries':{}}
for w in [2,4]:
 values=s['comparisons'][str(w)]
 out['first_differences'][str(w)]=[x for x in values if not x['bit_identical']][:10]
 out['early_boundaries'][str(w)]=[x for x in values if int(x['owner'].rsplit('-',1)[1])<=3]
 entries=[]
 for v in values:
  if not v['owner'].startswith('ffn_moe_topk-'):continue
  arrays=[]
  for width,file in [(1,v['file1']),(w,v['filew'])]:
   d=ROOT/'chain-capture-output'/f'{c.MODE}-w{width}'
   t=next(json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines() if json.loads(l)['file']==file)
   arrays.append([int(x) for x in c.read_tensor(d,t,1)])
  a,b=arrays
  entries.append({'layer':int(v['owner'].rsplit('-',1)[1]),'scalar':a,'wide':b,'order_identical':a==b,'set_identical':sorted(a)==sorted(b)})
 out['routing'][str(w)]={'layers':entries,'first_order_difference':next((x['layer'] for x in entries if not x['order_identical']),None),'first_membership_difference':next((x['layer'] for x in entries if not x['set_identical']),None),'order_difference_layers':[x['layer'] for x in entries if not x['order_identical']],'membership_difference_layers':[x['layer'] for x in entries if not x['set_identical']]}
(ROOT/'boundary-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({w:{k:v for k,v in r.items() if k!='layers'} for w,r in out['routing'].items()},indent=2))
