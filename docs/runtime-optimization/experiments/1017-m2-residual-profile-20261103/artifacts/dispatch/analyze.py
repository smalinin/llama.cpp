from pathlib import Path
import json,re,collections
P=Path(__file__).resolve().parent;r=json.loads((P/'results.json').read_text());base=json.loads((P.parent/'results.json').read_text());assert r['complete']
v=r['variants'][0];assert v['complete']and v['exit_code']==0 and not v['floor_hit']and not v['warnings']and not v.get('forced_kill')
for q in v['requests']:
 ref=next(z for z in base['variants'][0]['requests']if z['n']==q['n']);assert q['tokens_sha256']==ref['tokens_sha256'];assert all(q['response']['timings'][k]==ref['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])
log=(P/'native/server.log').read_text(errors='replace');requests=[];cur=None;lastalloc=None;copies=[]
def fields(s):return {k:int(v)if re.fullmatch(r'-?\d+',v)else v for k,v in re.findall(r'(\w+)=([^ ]+)',s)}
for line in log.splitlines():
 if 'processing task, is_child'in line:
  cur={'task':int(re.search(r'task (\d+)',line)[1]),'rebuilds':[],'compute':[],'alloc_splits':[]};lastalloc=None
 if 'sched_copy: edge_mode_name='in line:
  m=re.search(r'edge_mode_name=(.*?) count=(\d+) bytes=(\d+) host_sync=(\d+)',line);edge=m[1].split('|',3);copies.append(dict(zip(['src','dst','mode','tensor'],edge),count=int(m[2]),bytes=int(m[3]),host_sync=int(m[4])))
 if cur is None:continue
 if 'prompt eval time ='in line:cur['n']=int(re.search(r'/\s*(\d+) tokens',line)[1])
 if 'sched_timing: event=alloc_graph'in line:lastalloc=fields(line.split('sched_timing:',1)[1])
 if 'sched_timing: event=alloc_splits'in line:cur['alloc_splits'].append(fields(line.split('sched_timing:',1)[1]))
 if 'sched_timing: event=compute_splits'in line:cur['compute'].append(fields(line.split('sched_timing:',1)[1]))
 if 'graph_timing: event=rebuild'in line:
  x=fields(line.split('graph_timing:',1)[1]);assert lastalloc is not None and x['nodes']==lastalloc['input_nodes'];x['scheduler_alloc_total_us']=lastalloc['total_us'];x['alloc_outside_scheduler_timer_us']=x['alloc_us']-lastalloc['total_us'];cur['rebuilds'].append(x);lastalloc=None
 if 'stop processing: n_tokens'in line:requests.append(cur);cur=None
assert len(requests)==3
out={'complete':True,'request_count':3,'outputs_counters_match_native':True,'scope':'Diagnostic counters across entire requests including16 outputs; graph node-count buckets are specific to observed DS41/dflash graphs, not a generic classifier. alloc_outside_scheduler_timer includes input-copy wait before the sched timer and call/log overhead, not a direct wait-only timer. Copy stats cumulative across target/draft and all3requests.' ,'requests':[]}
for q in requests:
 hist=collections.Counter(x['nodes']for x in q['rebuilds']);t=[x for x in q['rebuilds']if x['nodes']>1000 and x['tokens']>6];out['requests'].append({'n':q['n'],'rebuild_node_histogram':dict(hist),'target_wide_rebuild_count':len(t),'target_wide_rebuild_ms':{k:sum(x[k]for x in t)/1000 for k in ['build_us','alloc_us','total_us','scheduler_alloc_total_us','alloc_outside_scheduler_timer_us']},'all_compute_ms':{k:sum(x[k]for x in q['compute'])/1000 for k in ['input_wait_us','input_copy_us','total_us']},'alloc_splits_count':len(q['alloc_splits']),'alloc_splits_realloc_count':sum(x['reallocated']for x in q['alloc_splits']),'alloc_splits_sync_ms':sum(x['sync_us']for x in q['alloc_splits'])/1000})
by=collections.defaultdict(lambda:dict(count=0,bytes=0,host_sync=0))
for q in copies:
 z=by[q['mode']]
 for k in ['count','bytes','host_sync']:z[k]+=q[k]
out['copy_modes']=dict(by);out['largest_copy_edges']=sorted(copies,key=lambda x:-x['bytes'])[:25];out['copy_edge_count']=len(copies)
mem=[z['min_free']for phase in ['load','inference']for z in v['memory'][phase].values()if isinstance(z,dict)and'min_free'in z];out['all_run_min_free_mib']=min(mem)/1048576;assert min(mem)>=512*1048576 and not v['memory'].get('error')
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
