from pathlib import Path
import json,runpy
R=Path(__file__).resolve().parent
H=runpy.run_path(str(R.parent/'stage19/analyze.py'));compare=H['compare'];logits=H['logits'];sha=H['sha'];MODE=H['MODE']
D=R/'model-output'
result={'windows':{},'controls':{}}
hp=json.loads((D/'model-hparams.json').read_text())
for name,length,widths in [('k4096',3033,[1,4]),('ratio1',3289,[1,2,3,4]),('ratio2',6617,[1,2,3,4])]:
 out={};ref=logits(D/name,1)
 for w in widths:
  p=logits(D/name,w);v=compare(ref,p)
  rows=[json.loads(l) for l in (D/name/f'{MODE}-w{w}-trace.jsonl').read_text().splitlines()]
  assert len(rows)==95*3
  transitions={};point_rows=[]
  for layer in [0,2,20]:
   previous=None;points=[]
   for r in rows:
    if r['layer']!=layer:continue
    ratio=hp['compress_ratios'][layer];pos=r['absolute_position']
    pad=lambda x:max(256,((x+255)//256)*256)
    raw=min(r['raw_limit'],pad(pos+1));comp=pad((pos+1)//ratio) if ratio else 0
    nk=raw+comp
    assert nk<=r['raw']+r['compressed']
    source_nk=r['raw']+r['compressed'];nkv=r['n_kv_max']
    sparse=bool(nkv>0 and nk>=max(4096,2*nkv))
    source_sparse=bool(nkv>0 and source_nk>=max(4096,2*nkv))
    item={**r,'query_nk':nk,'source_nk':source_nk,'query_sparse_from_predicate':sparse,'source_sparse_from_predicate':source_sparse}
    state=(nk,nkv,sparse,source_nk,source_sparse)
    if state!=previous:points.append(item);previous=state
    if pos in [3070,3071,3072,3073,3326,3327,3328,3329,6655,6656,6657,6658]:point_rows.append(item)
   transitions[str(layer)]=points
  v.update(prompt_tokens=length,prompt_sha256=sha(D/name/'prompt.i32'),forced_sha256=sha(D/name/'forced.i32'),trace_rows=len(rows),transitions=transitions,boundary_rows=point_rows)
  out[str(w)]=v
 result['windows'][name]=out
result['controls']['trace-ratio2-w4']=compare(logits(D/'ratio2',4),logits(D/'ratio2-no-trace',4))
for w in [1,4]:
 p=logits(D/'baseline',w)
 result['controls'][f'baseline95-w{w}']={'sha256':sha(p),'expected_sha256':'188c8690abade05294ac66643c56bc7727b21adb3ad6a65b5061e6ca0dfaed5b'}
assert result['controls']['trace-ratio2-w4']['bit_identical']
assert all(v['sha256']==v['expected_sha256'] for k,v in result['controls'].items() if k.startswith('baseline'))
(R/'model-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'windows':{n:{w:{k:v for k,v in c.items() if k not in ['transitions','boundary_rows']} for w,c in ws.items()} for n,ws in result['windows'].items()},'controls':result['controls']},indent=2))
