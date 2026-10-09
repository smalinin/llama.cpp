#!/usr/bin/env python3
from array import array
import math
import hashlib
import json
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'model-output'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(line) for line in p.read_text().splitlines()]
def compare_rows(reference,actual,vocab,count):
 first=None;different=0;maximum=total=0.0
 with reference.open('rb') as rf,actual.open('rb') as af:
  for row in range(count):
   rb=rf.read(vocab*4);ab=af.read(vocab*4)
   if rb==ab:continue
   different+=1;x=array('f');x.frombytes(rb);y=array('f');y.frombytes(ab)
   delta=[float(u)-float(v) for u,v in zip(x,y)]
   local_max=max(map(abs,delta));squares=sum(d*d for d in delta)
   maximum=max(maximum,local_max);total+=squares
   if first is None:first={'row':row,'first_vocab':next(i for i in range(vocab) if rb[4*i:4*i+4]!=ab[4*i:4*i+4]),'max_abs':local_max,'rms':math.sqrt(squares/vocab)}
 return {'first_difference':first,'different_rows':different,'max_abs':maximum,'rms':math.sqrt(total/(count*vocab))}
a=D/'cache-history-w1';b=D/'cache-history-w4'
ca=read(a/'counts.json');cb=read(b/'counts.json')
assert ca['rows']==cb['rows']==1024 and ca['vocab']==cb['vocab']
assert all((p/'logits.f32').stat().st_size==1024*ca['vocab']*4 for p in [a,b])
reference=sha(a/'logits.f32');actual=sha(b/'logits.f32')
scalar={(r['pos'],r['layer']):r for r in lines(a/'raw-trace.jsonl')}
wide=lines(b/'raw-trace.jsonl');raw=[r for r in wide if r['layer']==0]
physical=all(all(r[k]==scalar[(r['pos'],r['layer'])][k] for k in ['raw','physical_index','effective_total']) for r in wide)
results={}
for width in [3,4]:
 directory=D/f'cache-history-w{width}'
 trace=lines(directory/'raw-trace.jsonl');raw_width=[x for x in trace if x['layer']==0]
 results[str(width)]={'bit_identical':reference==sha(directory/'logits.f32'),'physical_trace_identical_to_scalar':all(all(r[k]==scalar[(r['pos'],r['layer'])][k] for k in ['raw','physical_index','effective_total']) for r in trace),'counts':read(directory/'counts.json'),'sha256':sha(directory/'logits.f32'),'raw_crops_at_observed_layers':sum(r['raw']<r['source_raw'] for r in trace),'transitions':[r for i,r in enumerate(raw_width) if not i or r['raw']!=raw_width[i-1]['raw']]}
metrics=compare_rows(a/'logits.f32',D/'cache-history-w3/logits.f32',ca['vocab'],1024)
results['3']['metrics']=metrics
results['4']['metrics']=metrics if results['3']['sha256']==results['4']['sha256'] else compare_rows(a/'logits.f32',b/'logits.f32',ca['vocab'],1024)
ratio2=[r for r in lines(a/'raw-trace.jsonl') if r['layer']==2]
sparse=lambda r:r['n_kv_max']>0 and r['effective_total']>=max(4096,2*r['n_kv_max'])
out={'ratio2_dispatch_changes':[dict(r,sparse=sparse(r)) for i,r in enumerate(ratio2) if not i or sparse(r)!=sparse(ratio2[i-1])],'rows':1024,'prompt_tokens':ca['prompt_tokens'],'widths':[1,3,4],'comparisons':results,
 'bit_identical':all(v['bit_identical'] for v in results.values()),'reference_sha256':reference,'actual_sha256':actual,
 'wide_3_4_bit_identical':results['3']['sha256']==results['4']['sha256'],'argmax_identical':all(u['argmax']==v['argmax'] for u,v in zip(lines(a/'rows.jsonl'),lines(b/'rows.jsonl'))),
 'physical_trace_identical_to_scalar':physical and all(v['physical_trace_identical_to_scalar'] for v in results.values()),'effective_extents':sorted({r['raw'] for r in raw}),
 'source_extents':sorted({r['source_raw'] for r in raw}),
 'first_position':raw[0]['pos'],'last_position':raw[-1]['pos'],
 'transitions':[r for i,r in enumerate(raw) if not i or r['raw']!=raw[i-1]['raw']],
 'ring_wraps':[{'before':raw[i-1],'after':r} for i,r in enumerate(raw) if i and r['physical_index']<raw[i-1]['physical_index']],
 'scope':'Fixed Stage19 token IDs on the restored Stage21 ratio2 prompt; compare scalar and wide numerics, not free-answer quality.'}
(R/'model-long-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
