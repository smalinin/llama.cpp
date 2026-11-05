from pathlib import Path
import json,re,sys
sys.setrecursionlimit(20000)
p=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts');pair=[]
for flag in ['0','1']:
 for l in(p/('metadata-'+flag)/'graphs.jsonl').open():
  j=json.loads(l)
  if'graph'in j:pair.append(j['graph']);break
A,B=[{x['id']:x for x in g['tensors']}for g in pair]
def find(ts):
 return next(t for t in ts.values()if t['op']=='SET_ROWS'and t['root_name']=='cache_k_l25'and t['ne'][1]==768)
a,b=find(A)['src'][0],find(B)['src'][0];seen=set();diffs=[]
def walk(x,y,path):
 if not x or not y:
  if x!=y:diffs.append((path,'NULL'))
  return
 if(x,y)in seen:return
 seen.add((x,y));a,b=A[x],B[y]
 for key in ['op','type','ne','nb','view_offs','op_params_i32']:
  if a[key]!=b[key]:diffs.append((path,key,a['name'],b['name'],a[key],b[key]));return
 if a['op']=='NONE':
  if a['name']!=b['name']:diffs.append((path,'leafname',a['name'],b['name'],a['input'],b['input']))
  return
 for i,(x,y)in enumerate(zip(a['src'],b['src'])):
  if a['op']=='CPY'and i==1:continue
  if a['op']=='SET_ROWS'and i==2 and (x==a['id']or y==b['id']):continue
  walk(x,y,path+'/'+a['op']+':'+a['name']+':'+str(i))
walk(a,b,'rawK25');print('differences',len(diffs));print(json.dumps(diffs[:8],indent=2))
