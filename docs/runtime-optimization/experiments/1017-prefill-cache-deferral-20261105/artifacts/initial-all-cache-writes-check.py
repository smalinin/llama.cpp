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
arms={n:collect(n)for n in ['owner-default','owner-on']};assert len(arms['owner-default'])==len(arms['owner-on'])==9
for off,on in zip(*arms.values()):
 assert(off['pos'],off['tokens'],off['kv_only'])==(on['pos'],on['tokens'],on['kv_only'])
 def semantic(r):return sorted((w['root'],w['type'],tuple(w['ne']),w['root_bytes'],w['backend'],tuple(w['value_ne']),tuple(w['indices_ne']))for w in r['writes'])
 assert semantic(off)==semantic(on),(off['pos'],semantic(off),semantic(on))
 assert len(on['writes'])==len({w['root']for w in on['writes']}),(on['pos'],'duplicate root write')
results=[json.loads((P/n/'counts-results.json').read_text())for n in arms];ref=results[0]['requests'][0];same=all(x['tokens_sha256']==ref['tokens_sha256']and(x['response']['timings']['draft_n'],x['response']['timings']['draft_n_accepted'])==(ref['response']['timings']['draft_n'],ref['response']['timings']['draft_n_accepted'])for r in results for x in r['requests']);assert same
j=dict(complete=True,description='Actual backend graph snapshots: all persistent Q8 CSA/LID cache SET_ROWS, not only source20 CSA. No numeric data read. Original per-arm observer manifests unchanged.',selected_ubatches_per_arm=9,all_ten_request_outputs_counters_exact=same,per_chunk_write_count=[len(r['writes'])for r in arms['owner-on']],total_writes_per_arm=sum(len(r['writes'])for r in arms['owner-on']),same_root_type_shape_size_backend_value_shape_index_shape=True,one_write_per_cache_root_per_ubatch=True,all_present_first_readers_after_writes=True,arms=arms)
(P/'all-cache-writes-checks.json').write_text(json.dumps(j,indent=2)+'\n');print({k:v for k,v in j.items()if k!='arms'})
