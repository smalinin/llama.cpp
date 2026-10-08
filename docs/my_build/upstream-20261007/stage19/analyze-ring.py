from pathlib import Path
import json,math,struct
R=Path(__file__).resolve().parent
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
D=R/'boundary-output'
def intervals(indices):
 result=[]
 for i in indices:
  if result and result[-1][1]+1==i:result[-1][1]=i
  else:result.append([i,i])
 return result
summary={'native_extent_transitions':{},'ring_masks':{}}
native=[json.loads(l) for l in (D/'legacy'/f'{MODE}-w1-trace.jsonl').read_text().splitlines()]
for layer in [0,2,20]:
 previous=None;changes=[]
 for r in native:
  if r['layer']!=layer:continue
  shape=(r['raw'],r['compressed'])
  if shape!=previous:changes.append(r);previous=shape
 summary['native_extent_transitions'][str(layer)]=changes
for variant,widths in [('legacy',[1,3,4]),('general',[1,2,3,4])]:
 for w in widths:
  out={}
  for pos in [765,766,767,768,769,770]:
   target=pos-23;start=target//w*w;col=target-start
   d=D/variant/f'{MODE}-w{w}-batch{start}'
   ts=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
   mask=next(t for t in ts if t['owner']=='FA-0' and t['role']=='src3')
   raw=(d/mask['file']).read_bytes()
   values=struct.unpack_from('<'+'e'*mask['ne'][0],raw,col*mask['nb'][1])
   visible=[i for i,x in enumerate(values) if math.isfinite(x)]
   assert len(visible)==128
   out[str(pos)]={'input':target,'batch_start':start,'column':col,'raw_rows':len(values),'visible_count':len(visible),'physical_intervals':intervals(visible)}
  summary['ring_masks'][f'{variant}-w{w}']=out
(R/'ring-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'native_extent_transitions':{k:[{'position':r['absolute_position'],'raw':r['raw'],'compressed':r['compressed']} for r in rows] for k,rows in summary['native_extent_transitions'].items()},'native_ring':summary['ring_masks']['legacy-w1']},indent=2))
