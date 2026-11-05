from pathlib import Path
import json,collections
P=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts')

def collect(folder):
 f=P/folder/'scopes.jsonl';h={}
 for line in f.open():
  r=json.loads(line);h[r['id']]={k:r[k]for k in ['kind','parent']}
 def top(r):
  while r['parent']in h:r=h[r['parent']]
  return r['kind']
 out=[]
 for line in f.open():
  r=json.loads(line)
  if r['kind']!='ubatch'or top(r)!='target_decode':continue
  graphs=sorted((g for g in r['calls']if g['kind']=='graph_compute'),key=lambda x:x['start'])
  if not any('graph_snapshot'in g for g in graphs):continue
  writes=[];readers={}
  for gi,g in enumerate(graphs):
   snap=g.get('graph_snapshot')
   if not snap:continue
   tensors={t['id']:t for t in snap['tensors']}
   for ni,nid in enumerate(snap['nodes']):
    t=tensors[nid]
    if t['op']=='SET_ROWS'and t['root_name'].startswith('cache_k_l'):
     v=tensors[t['src'][0]];ix=tensors[t['src'][1]]
     writes.append(dict(root=t['root_name'],type=t['type'],ne=t['ne'],root_bytes=t['root_bytes'],backend=g['backend'],index=[gi,ni],name=t['name'],value_ne=v['ne'],indices_ne=ix['ne']))
    elif t['op']not in ['NONE','VIEW','RESHAPE','PERMUTE','TRANSPOSE','SET_ROWS']:
     for src in t['src']:
      if src and tensors[src]['root_name'].startswith('cache_k_l'):
       key=tensors[src]['root_name']
       if key not in readers:readers[key]=dict(index=[gi,ni],name=t['name'],backend=g['backend'])
  for w in writes:
   if w['root']in readers:assert w['index']<readers[w['root']]['index'],(folder,r['pos'],w,readers[w['root']])
  out.append(dict(pos=r['pos'],tokens=r['tokens'],kv_only=r['kv_only'],writes=writes,first_readers=readers))
 return out
rows=collect('owner-on')
for r in rows[:1]:
 for w in r['writes']:print(w)
