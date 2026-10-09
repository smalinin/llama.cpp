#!/usr/bin/env python3
from array import array
import ctypes,hashlib,json,math,runpy
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'model-output'
helpers=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))
read_tensor,compare=helpers['read_tensor'],helpers['compare']
metric=helpers['metric']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def full_compare(a,b):
 ca,cb=read(a/'counts.json'),read(b/'counts.json')
 assert ca['rows']==cb['rows']==1024 and ca['vocab']==cb['vocab']
 vocab=ca['vocab'];first=None;different=0;maximum=total=0.0
 assert all((d/'logits.f32').stat().st_size==1024*vocab*4 for d in [a,b])
 with (a/'logits.f32').open('rb') as af,(b/'logits.f32').open('rb') as bf:
  for row in range(1024):
   ab,bb=af.read(vocab*4),bf.read(vocab*4)
   if ab==bb:continue
   different+=1;x=array('f');x.frombytes(ab);y=array('f');y.frombytes(bb)
   out=(ctypes.c_double*5)();metric(x.buffer_info()[0],y.buffer_info()[0],vocab,0,out);assert out[4]==0
   maximum=max(maximum,out[2]);total+=out[0]
   if first is None:first={'output_index':row,'query_position':ca['prompt_tokens']+row-1,'max_abs':out[2],'rms':math.sqrt(out[0]/vocab)}
 return {'bit_identical':different==0,'first_difference':first,'different_rows':different,'max_abs':maximum,'rms':math.sqrt(total/(1024*vocab)),
  'argmax_identical':all(x['argmax']==y['argmax'] for x,y in zip(lines(a/'rows.jsonl'),lines(b/'rows.jsonl')))}
result={'controls':{},'full_comparisons':{},'captures':{}}
for width in [1,3,4]:
 label=f'cache-history-w{width}';new=D/label/'logits.f32';old=R.parent/'stage22/model-output'/label/'logits.f32'
 result['controls'][str(width)]={'sha256':sha(new),'reference_sha256':sha(old),'bit_identical':sha(new)==sha(old)}
assert all(v['bit_identical'] for v in result['controls'].values()),'capture perturbs frozen logits'
for kind,widths in [('cache',[3,4]),('fresh',[4])]:
 for width in widths:result['full_comparisons'][f'{kind}-w{width}']=full_compare(D/f'{kind}-history-w1',D/f'{kind}-history-w{width}')
result['fresh_sha256']={str(w):sha(D/f'fresh-history-w{w}/logits.f32') for w in [1,4]}
for kind,widths in [('cache',[1,3,4]),('fresh',[1,4])]:
 for target in [749,750]:
  dirs={w:D/f'{kind}-history-w{w}'/f'batch{target//w*w}' for w in widths}
  align={w:read(d/'alignment.json') for w,d in dirs.items()}
  tensors={w:lines(d/'tensors.jsonl') for w,d in dirs.items()}
  chain={w:[t for t in ts if t['role']=='output'] for w,ts in tensors.items()}
  assert len(set(map(len,chain.values())))==1
  records={str(w):[] for w in widths[1:]};occ={}
  for i,a in enumerate(chain[1]):
   key=(a['owner'],a['role']);index=occ.get(key,0);occ[key]=index+1
   for w in widths[1:]:
    b=chain[w][i];assert key==(b['owner'],b['role'])
    axes=[j for j,(x,y) in enumerate(zip(a['ne'],b['ne'])) if x!=y];assert len(axes)<=1,(a,b)
    axis=axes[0] if axes else None
    if axis is not None:assert a['ne'][axis]==1 and b['ne'][axis]==w,(a,b)
    aa=read_tensor(dirs[1],a,axis,0);bb=read_tensor(dirs[w],b,axis,target-align[w]['start'] if axis is not None else 0)
    v=compare(aa,bb);v.update(owner=a['owner'],role=a['role'],occurrence=index,axis=axis,elements=len(aa),op=a['op'],file1=a['file'],filew=b['file']);records[str(w)].append(v)
  result['captures'][f'{kind}-{target}']={'alignment':align,'counts':{w:len(c) for w,c in chain.items()},'chain':records,'first_differences':{w:[x for x in xs if not x['bit_identical']][:15] for w,xs in records.items()}}
(R/'capture-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v if k!='captures' else {t:{a:b for a,b in z.items() if a!='chain'} for t,z in v.items()} for k,v in result.items()},indent=2))
