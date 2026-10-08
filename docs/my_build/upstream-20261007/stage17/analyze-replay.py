from array import array
import hashlib,json,runpy
from pathlib import Path

R=Path(__file__).resolve().parent
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
compare=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['compare']
def load(p):
    a=array('f');a.frombytes(p.read_bytes());return a
reference=R.parent/'stage16/compressor-candidate-output'/f'{MODE}-w1-logits.f32'
base=load(reference);assert len(base)==95*129280
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={}
for w in [1,2,3,4]:
    p=R/'replay-output'/f'{MODE}-w{w}-logits.f32';a=load(p)
    rows=[json.loads(l) for l in (p.parent/f'{MODE}-w{w}-rows.jsonl').read_text().splitlines()]
    assert len(rows)==95 and all(r['argmax']==r['reference'] for r in rows)
    value=compare(base,a);value.update(sha256=sha(p),reference_sha256=sha(reference),all95_argmax_native=True)
    assert value['bit_identical'] and value['sha256']==value['reference_sha256'],(w,value)
    result[str(w)]=value
(R/'replay-control-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
