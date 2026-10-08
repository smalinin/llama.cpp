from pathlib import Path
import hashlib,json,runpy
from array import array
R=Path(__file__).resolve().parent
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s={}
for label,refdir,ws in [('v1-layer20',R/'candidate-output',[1,3]),('v2-explain',R.parent/'stage17/explain-replay-output',[1,2,3,4]),('v2-baseline',R.parent/'stage17/replay-output',[1,2,3,4])]:
 v={}
 for w in ws:
  p=R/'v2-output'/label/f'{MODE}-w{w}-logits.f32'
  ref=refdir/f'{MODE}-w{w if label=="v1-layer20" else 1}-logits.f32'
  a=array('f');a.frombytes(ref.read_bytes());b=array('f');b.frombytes(p.read_bytes())
  c=compare(a,b);rows=list(map(json.loads,p.with_name(p.name.replace('-logits.f32','-rows.jsonl')).read_text().splitlines()))
  c.update(sha256=sha(p),reference_sha256=sha(ref),rows=len(rows),argmax_differences=[r['index'] for r in rows if r['argmax']!=r['reference']])
  v[str(w)]=c
 s[label]=v
assert all(v['bit_identical'] for v in s['v1-layer20'].values()),'layer20 capture changed v1'
(R/'v2-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
