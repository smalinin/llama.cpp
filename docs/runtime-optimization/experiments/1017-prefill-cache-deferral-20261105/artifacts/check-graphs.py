from pathlib import Path
import json,hashlib,collections,re,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts'
def analyze(g):
 ts={t['id']:t for t in g['tensors']};memo={};active=set()
 def sig(n):
  if not n:return''
  if n in memo:return memo[n]
  assert n not in active,'DAG cycle';active.add(n);t=ts[n];d=[t['op'],t['type'],t['ne'],t['nb'],t['view_offs'],t['op_params_i32']]
  if t['op']=='NONE':
   name=re.sub(r'^CUDA\d+#(.*)#\d+$',r'\1',t['name']);d+=['input'if t['input']else'leaf',name]
  else:
   src=[]
   for i,k in enumerate(t['src']):
    if not k:continue
    if t['op']=='FILL'and i==0 and ts[k]['op']=='NONE'and not ts[k]['input']and re.fullmatch(r'leaf_\d+',ts[k]['name']):
     a=ts[k];src.append(['fill_shape',a['type'],a['ne'],a['nb']])
    elif (t['op']=='CPY'and i==1)or(t['op']=='SET_ROWS'and i==2 and k in active):
     a=ts[k];name=re.sub(r'^node_\d+$','anonymous',a['root_name']);src.append(['destination',name,a['type'],a['ne'],a['nb'],a['root_offset']])
    else:src.append(sig(k))
   d+=[src]
  h=hashlib.sha256(json.dumps(d,separators=(',',':')).encode()).hexdigest();memo[n]=h;active.remove(n);return h
 writes=[];ops=collections.Counter();order={nid:i for i,nid in enumerate(g['nodes'])}
 for nid in g['nodes']:
  t=ts[nid];ops[(t['op'],sig(nid))]+=1
  if t['op']!='SET_ROWS':continue
  dst=ts[t['src'][2]];key=(re.sub(r'^node_\d+$','anonymous',dst['root_name']),dst['type'],tuple(dst['ne']),dst.get('buffer_owner'))
  writes.append({'destination':key,'signature':(sig(t['src'][0]),sig(t['src'][1]),sig(t['src'][2])),'index':order[nid],'name':t['name'],'id':nid,'root':t['root'],'source':ts[t['src'][0]]['name']})
 first={}
 for nid in g['nodes']:
  t=ts[nid]
  if t['op']in ['NONE','VIEW','RESHAPE','PERMUTE','TRANSPOSE','SET_ROWS']:continue
  for src in t['src']:
   if src:first.setdefault(ts[src]['root'],{'index':order[nid],'name':t['name'],'op':t['op']})
 for w in writes:
  if w['destination'][0].startswith('cache_k_l')and w['destination'][2][1]>2000:
   w['first_reader']=first.get(w['root']);w['before_reader']=w['first_reader']is None or w['index']<w['first_reader']['index']
 return{'nodes':len(g['nodes']),'ops':ops,'writes':writes}
rows=[]
for flag in ['0','1']:
 gs=[json.loads(l)for l in(P/('metadata-'+flag)/'graphs.jsonl').open()];rows.append([x for x in gs if'graph'in x])
assert len(rows[0])==len(rows[1])==10
out={'complete':False,'groups':[]}
for a,b in zip(*rows):
 assert(a['tokens'],a['node_count'])==(b['tokens'],b['node_count'])
 x,y=analyze(a['graph']),analyze(b['graph']);canon=lambda z:collections.Counter((tuple(w['destination']),tuple(w['signature']))for w in z['writes']);samewrites=canon(x)==canon(y);sameops=x['ops']==y['ops'];readok=all(w.get('before_reader',True)for w in y['writes'])
 xa={(tuple(w['destination']),tuple(w['signature'])):w for w in x['writes']};changes=[{'destination':w['destination'],'old_index':xa[(tuple(w['destination']),tuple(w['signature']))]['index'],'new_index':w['index'],'first_reader':w.get('first_reader'),'new_name':w['name']}for w in y['writes']if(tuple(w['destination']),tuple(w['signature']))in xa and xa[(tuple(w['destination']),tuple(w['signature']))]['index']!=w['index']]
 group={'tokens':a['tokens'],'nodes':a['node_count'],'off_deferred':a['deferred'],'on_deferred':b['deferred'],'all_write_operands_and_counts_exact':samewrites,'all_semantic_operator_multisets_exact':sameops,'compressed_writes_before_first_reader':readok,'writes_off':len(x['writes']),'writes_on':len(y['writes']),'moved_writes':changes};out['groups'].append(group)
 if not samewrites:group['write_mismatches']={'off_only':[str(k)for k in(canon(x)-canon(y))],'on_only':[str(k)for k in(canon(y)-canon(x))]}
 if not sameops:group['operator_mismatch_counts']={'off_only':sum((x['ops']-y['ops']).values()),'on_only':sum((y['ops']-x['ops']).values())}
 print('GRAPH',a['tokens'],a['node_count'],'writes',samewrites,'ops',sameops,'reader',readok,'moved',len(changes),flush=True)
out['complete']=all(x['all_write_operands_and_counts_exact']and x['all_semantic_operator_multisets_exact']and x['compressed_writes_before_first_reader']for x in out['groups']);(P/'graph-checks.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2('/tmp/check-deferral-graphs.py',P/'check-graphs.py');assert out['complete'],'Graph audit mismatch; inspect graph-checks.json'
