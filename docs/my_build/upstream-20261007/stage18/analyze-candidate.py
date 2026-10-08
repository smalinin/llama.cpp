from pathlib import Path
from array import array
import hashlib,json,runpy
R=Path(__file__).resolve().parent
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s={}
for label,reference in [('candidate',R.parent/'stage17/explain-replay-output'),('baseline',R.parent/'stage17/replay-output')]:
 d=R/f'{label}-output'
 if not (d/f'{MODE}-w4-logits.f32').exists():continue
 ref=reference/f'{MODE}-w1-logits.f32';a=array('f');a.frombytes(ref.read_bytes())
 v={}
 for w in [1,2,3,4]:
  p=d/f'{MODE}-w{w}-logits.f32';b=array('f');b.frombytes(p.read_bytes());c=compare(a,b)
  rows=list(map(json.loads,(d/f'{MODE}-w{w}-rows.jsonl').read_text().splitlines()))
  c.update(sha256=sha(p),scalar_reference_sha256=sha(ref),rows=len(rows),argmax_differences=[r['index'] for r in rows if r['argmax']!=r['reference']])
  v[str(w)]=c
 s[label]=v
(R/'candidate-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
