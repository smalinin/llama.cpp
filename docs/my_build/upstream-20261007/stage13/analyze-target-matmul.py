#!/usr/bin/env python3
from array import array
import importlib.util
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',ROOT/'analyze-chain.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
D=ROOT/'target-matmul-output';OLD=ROOT.parent/'stage12/target-ffn-output'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 a=array('f');a.frombytes(p.read_bytes());return a
summary={'rows':65,'controls':{},'variants':{},'replacement_counts':{},'first_layer2_hidden':{}}
for mode in ['decode-scalar-fa-upgate-router','decode-scalar-fa-upgate-hc-router','decode-scalar-fa-upgate-float2d']:
 base=read(D/f'{mode}-w1-logits.f32')
 assert len(base)==65*129280
 p=D/f'{mode}-w1-logits.f32';ref=OLD/'barrier-w1-logits.f32'
 summary['controls'][mode]={'sha256':sha(p),'reference_sha256':sha(ref),'bit_identical':sha(p)==sha(ref)}
 rows1=[json.loads(l) for l in (D/f'{mode}-w1-rows.jsonl').read_text().splitlines()]
 for w in [1,2,4]:
  label=f'{mode}-w{w}';p=D/f'{label}-logits.f32';a=read(p)
  rows=[json.loads(l) for l in (D/f'{label}-rows.jsonl').read_text().splitlines()];assert len(rows)==65
  result=c.compare(base,a);result.update(sha256=sha(p),argmax_difference_indices=[i for i,(x,y) in enumerate(zip(rows1,rows)) if x['argmax']!=y['argmax']],index19_verify_minus_check=a[19*129280+23393]-a[19*129280+4085])
  summary['variants'][label]=result
  counts=json.loads((D/f'{label}-matmul-counts.json').read_text())
  prefixes={}
  for key,count in counts.items():prefix=key.rsplit('-',1)[0];prefixes[prefix]=prefixes.get(prefix,0)+count
  assert all(count==(128 if key.startswith('hc_mixes-') else 64) for key,count in counts.items() if not key.startswith('node_'))
  assert all(count in [32,64] for key,count in counts.items() if key.startswith('node_'))
  if w==1:assert not counts
  else:
   assert prefixes['ffn_moe_logits']==2560
   if mode.endswith(('hc-router','float2d')):assert prefixes['hc_mixes']==5120
  summary['replacement_counts'][label]={'by_prefix':prefixes,'total':sum(counts.values()),'tensor_names':len(counts)}
  before=read(D/f'{label}-ffn_moe_swiglu_limited-2-before.f32')[:2304*6];after=read(D/f'{label}-ffn_moe_swiglu_limited-2-after.f32')[:2304*6]
  scalar=read(D/f'{mode}-w1-ffn_moe_swiglu_limited-2-after.f32')
  summary['first_layer2_hidden'][label]={'before':c.compare(scalar,before),'after':c.compare(scalar,after)}
assert all(x['bit_identical'] for x in summary['controls'].values())
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'controls':summary['controls'],'variants':summary['variants'],'counts':summary['replacement_counts']},indent=2))
