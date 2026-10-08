from pathlib import Path
import hashlib,json,runpy
R=Path(__file__).resolve().parent
helpers=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))
read,compare=helpers['read_tensor'],helpers['compare']
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
D=R/'capture-output'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'controls':{},'targets':{}}
for w in [1,2,3,4]:
 p=D/f'{MODE}-w{w}-logits.f32';ref=R.parent/'stage17/explain-replay-output'/p.name
 result['controls'][str(w)]={'sha256':sha(p),'reference_sha256':sha(ref),'bit_identical':sha(p)==sha(ref)}
assert all(v['bit_identical'] for v in result['controls'].values()),'capture changes logits'
for target in [231,232]:
 dirs={}
 align={}
 for w in [1,2,3,4]:
  start=target//w*w
  dirs[w]=D/f'{MODE}-w{w}-batch{start}'
  align[w]=json.loads((dirs[w]/'alignment.json').read_text())
  align[w]['column']=target-start
  align[w]['input_index']=target;align[w]['output_index']=target+1
  align[w]['absolute_position']=23+target
 tensors={w:[json.loads(l) for l in (dirs[w]/'tensors.jsonl').read_text().splitlines()] for w in dirs}
 chain={w:[t for t in ts if t['role']=='output' and not t['owner'].startswith('FA-')] for w,ts in tensors.items()}
 records={str(w):[] for w in [2,3,4]}
 for i,a in enumerate(chain[1]):
  for w in [2,3,4]:
   b=chain[w][i];assert (a['owner'],a['role'])==(b['owner'],b['role'])
   axes=[j for j,(x,y) in enumerate(zip(a['ne'],b['ne'])) if x!=y]
   assert len(axes)<=1,(a,b)
   axis=axes[0] if axes else None
   aa=read(dirs[1],a,axis,0)
   bb=read(dirs[w],b,axis,align[w]['column'] if axis is not None else 0)
   v=compare(aa,bb);v.update(owner=a['owner'],axis=axis,elements=len(aa),op=a['op'],file1=a['file'],filew=b['file'])
   records[str(w)].append(v)
 result['targets'][str(target)]={'alignment':align,'counts':{w:len(c) for w,c in chain.items()},'chain':records,
  'first_differences':{w:[x for x in xs if not x['bit_identical']][:15] for w,xs in records.items()}}
(R/'capture-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'controls':result['controls'],'targets':{t:{k:v for k,v in r.items() if k!='chain'} for t,r in result['targets'].items()}},indent=2))
