#!/usr/bin/env python3
from pathlib import Path
from array import array
import argparse,ctypes,hashlib,json,math,runpy
R=Path(__file__).resolve().parent;D=R/'model-output'
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
metric=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['metric']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def compare(a,b):
 ca,cb=read(a/'counts.json'),read(b/'counts.json');assert ca['rows']==cb['rows'] and ca['vocab']==cb['vocab']
 n,vocab=ca['rows'],ca['vocab'];first=None;count=0;maximum=total=0.0
 assert all((p/'logits.f32').stat().st_size==n*vocab*4 for p in [a,b])
 sa,sb=sha(a/'logits.f32'),sha(b/'logits.f32')
 if sa!=sb:
  with (a/'logits.f32').open('rb') as af,(b/'logits.f32').open('rb') as bf:
   for row in range(n):
    ab,bb=af.read(vocab*4),bf.read(vocab*4)
    if ab==bb:continue
    count+=1;x=array('f');x.frombytes(ab);y=array('f');y.frombytes(bb)
    out=(ctypes.c_double*5)();metric(x.buffer_info()[0],y.buffer_info()[0],vocab,0,out);assert out[4]==0
    maximum=max(maximum,out[2]);total+=out[0]
    if first is None:first={'output_index':row,'query_position':ca['prompt_tokens']+row-1,'max_abs':out[2],'rms':math.sqrt(out[0]/vocab)}
 ar,br=lines(a/'rows.jsonl'),lines(b/'rows.jsonl');assert len(ar)==len(br)==n
 return {'bit_identical':sa==sb,'reference_sha256':sa,'actual_sha256':sb,'rows':n,'first_difference':first,'different_rows':count,'max_abs':maximum,'rms':math.sqrt(total/(n*vocab)),
 'argmax_different_rows':[x['index'] for x,y in zip(ar,br) if x['argmax']!=y['argmax']]}
expected_layers=[2,8,14,20,24,28,32,36]
result={'controls':{},'histories':{},'counts':{},'indexer_selection':{},'physical_traces':{}}
completed={p.parent.name for p in D.glob('*/counts.json')}
for label in sorted(completed):
 c=read(D/label/'counts.json');result['counts'][label]=c
 counts=read(D/label/'indexer-counts.json');w=c['width'];q=c['rows']-1
 expected=sum(min(w,q-start) for start in range(0,q,w) if min(w,q-start)>1)
 required={f'blk.{layer}.indexer.proj.weight':expected for layer in expected_layers} if expected else {}
 result['indexer_selection'][label]={'columns_per_projection':expected,'counts':counts,'exact':counts==required};assert counts==required,(label,counts,required)
for label in ['baseline-w1','baseline-w4','sparse-control-w4','history-control-w4']:
 if label in completed:
  actual=sha(D/label/'logits.f32');reference=sha(R.parent/'stage22/model-output'/label/'logits.f32')
  result['controls'][label]={'sha256':actual,'reference_sha256':reference,'bit_identical':actual==reference}
for kind,parent in [('cache','stage22'),('fresh','stage23')]:
 label=f'{kind}-history-w1'
 if label in completed:
  actual=sha(D/label/'logits.f32');reference=sha(R.parent/parent/'model-output'/label/'logits.f32')
  result['controls'][label]={'sha256':actual,'reference_sha256':reference,'bit_identical':actual==reference}
  scalar={(r['pos'],r['layer']):r for r in lines(D/label/'raw-trace.jsonl')}
  for width in [2,3,4]:
   other=f'{kind}-history-w{width}'
   if other not in completed:continue
   result['histories'][other]=compare(D/label,D/other)
   trace=lines(D/other/'raw-trace.jsonl')
   result['physical_traces'][other]={'rows':len(trace),'exact':all(all(r[k]==scalar[(r['pos'],r['layer'])][k] for k in ['raw','physical_index','effective_total','n_kv_max']) for r in trace),'raw_extents':sorted({r['raw'] for r in trace if r['layer']==0})}
result['completed_cases']=sorted(completed);result['completed_count']=len(completed)
result['all_controls_exact']=all(c['bit_identical'] for c in result['controls'].values())
result['all_history_logits_exact']=bool(result['histories']) and all(v['bit_identical'] for v in result['histories'].values())
result['scope']='Fixed Stage19 token IDs on the Stage22 ratio2 prompt; arithmetic comparisons, not free-generation quality or production throughput.'
if not args.partial:assert len(completed)==12 and len(result['controls'])==6 and len(result['histories'])==6
path=R/('model-partial-summary.json' if args.partial else 'model-summary.json');path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['counts','indexer_selection']},indent=2))
