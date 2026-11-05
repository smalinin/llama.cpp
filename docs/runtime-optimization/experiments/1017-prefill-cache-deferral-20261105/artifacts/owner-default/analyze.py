from pathlib import Path
import json,collections,hashlib,gzip,statistics,shutil,subprocess
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts/owner-on';F=P/'scopes.jsonl'
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
headers={}
with F.open()as f:
 for l in f:
  x=json.loads(l);headers[x['id']]={k:x[k]for k in ['id','parent','kind','pos','tokens','kv_only']}
def root(x):
 while x['parent']in headers:x=headers[x['parent']]
 return x['kind']
summary=[];byop=collections.Counter();allcache=[];snapcount=0;eventowners={}
with F.open()as f:
 for l in f:
  row=json.loads(l)
  if row['kind']!='ubatch'or root(row)!='target_decode':continue
  calls=sorted(row['calls'],key=lambda x:x['start']);graphs=[x for x in calls if x['kind']=='graph_compute'];desc={};nodesprod={};details=[]
  for gi,c in enumerate(graphs):
   g=c.get('graph_snapshot')
   if not g:continue
   snapcount+=1;local={t['id']:t for t in g['tensors']};desc.update(local)
   for ni,nid in enumerate(g['nodes']):
    t=local[nid];byop[t['op']]+=1;nodesprod[nid]={'graph_index':gi,'node_index':ni,'backend':c['backend'],'name':t['name'],'op':t['op']}
    if t['op']!='SET_ROWS':continue
    value=local[t['src'][0]];indices=local[t['src'][1]];dst=local[t['src'][2]]
    copies=[y for y in calls if y['kind']in ['copy_iface','copy_sync']and y['start']<c['start']and y.get('dst',{}).get('root')==value['root']]
    origin=copies[-1]['src']if copies else None
    detail={'graph_index':gi,'node_index':ni,'backend':c['backend'],'name':t['name'],'id':nid,'root':t['root'],'root_data':t['root_data'],'root_buffer':t['root_buffer'],'root_bytes':t['root_bytes'],'cache_type':t['type'],'dst_owner':dst.get('buffer_owner'),'dst_usage':dst.get('buffer_usage'),'dst_ne':dst['ne'],'cache_root_name':t['root_name'],'value_name':value['name'],'value_root':value['root'],'value_owner':value.get('buffer_owner'),'value_root_data':value['root_data'],'value_root_bytes':value['root_bytes'],'value_shape':value['ne'],'value_bytes':value['bytes'],'indices_name':indices['name'],'indices_shape':indices['ne'],'indices_root':indices['root'],'indices_owner':indices.get('buffer_owner'),'copy_origin':origin,'copy_kind':copies[-1]['kind']if copies else None,'copy_accepted':copies[-1].get('async_accepted')if copies else None,'event_wait_before':[]}
    waits=[y for y in calls if y['kind']=='event_wait'and y['start']<c['start']and (not gi or y['start']>graphs[gi-1]['start'])]
    detail['event_wait_before']=[{'source':x['event_device'],'dest':x['backend'],'event':x['event'],'caller':hex(x['offset'])}for x in waits];details.append(detail)
  cache=[]
  for x in details:
   orig=x['copy_origin'];name=(orig['name']if orig else x['value_name'])
   if name.startswith('comp_kv_rot-20'):
    x['origin_name']=name;x['origin_owner']=orig.get('buffer_owner')if orig else x['value_owner'];x['origin_root']=orig['root']if orig else x['value_root'];x['origin_producer']=nodesprod.get(x['origin_root']);cache.append(x);allcache.append({'scope':row['id'],'pos':row['pos'],'kv_only':row['kv_only'],**x})
  events=[{'kind':x['kind'],'backend':x.get('backend'),'event_device':x.get('event_device'),'event':x.get('event'),'caller':hex(x['offset'])}for x in calls if x['kind']in['event_record','event_sync','event_wait']]
  records={}
  for x in events:
   if x['kind']=='event_record':records[x['backend']]=x['event'];eventowners[x['event']]=x['backend']
  scopes={'scope':row['id'],'pos':row['pos'],'tokens':row['tokens'],'kv_only':row['kv_only'],'graph_count':len(graphs),'snapshot_count':sum('graph_snapshot'in c for c in graphs),'descriptor_ids':len(desc),'set_rows':len(details),'source20_writes':cache,'events':events,'last_recorded_events':records,'graph_order':[{'index':i,'backend':c['backend'],'nodes':c['nodes'],'first':c.get('first',{}).get('name'),'last':c.get('last',{}).get('name')}for i,c in enumerate(graphs)]};summary.append(scopes)
r=json.loads((P/'counts-results.json').read_text());assert r['complete']and r['exit_code']==0 and not r['floor_breach']and not r['warnings']
ref=r['requests'][0]
exact=all(x['tokens_sha256']==ref['tokens_sha256']and(x['response']['timings']['draft_n'],x['response']['timings']['draft_n_accepted'])==(ref['response']['timings']['draft_n'],ref['response']['timings']['draft_n_accepted'])for x in r['requests']);assert exact
for f,h in r['config']['source_library_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
for f,h in r['config']['source_files'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
for x in r['config']['model_stats']:
 st=Path(x['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(x[k]for k in ['device','inode','bytes','mtime_ns'])
free={ph:{k:x['min_free']/2**20 for k,x in r['memory'][ph].items()if isinstance(x,dict)}for ph in ['load','inference']};assert min(v for x in free.values()for v in x.values())>=512
controls=[x for x in r['requests']if x['kind']in ['pre','post']];pp=statistics.mean(x['pp']for x in controls);armed=next(x['pp']for x in r['requests']if x['kind']=='counts')
j={'complete':True,'scope_headers':list(headers.values()),'target_ubatches':summary,'captured_graph_snapshots':snapcount,'ops_in_executed_snapshots':dict(byop),'source20_cache_writes':allcache,'native_control_pp8k':pp,'armed_pp8k':armed,'armed_delta_percent':100*(armed/pp-1),'all_five_outputs_draft_counters_exact_to_previous_default':exact,'runtime_source_libraries_unchanged':True,'target_shards_stat_unchanged':10,'actual_min_free_mib':free,'exit_code':0,'note':'Submission order and metadata only. Snapshots follow current graph pointers within each scope; numeric tensor data/row index values not read. Source copy correlation uses exact root identity in the same ubatch, not addresses across graphs. Timed armed request is observer-perturbed.'};save(P/'summary.json',j)
rawsha=hashlib.sha256();rawbytes=0
with F.open('rb')as fi,(P/'scopes.jsonl.gz').open('wb')as fo:
 with gzip.GzipFile(filename='',fileobj=fo,mode='wb',mtime=0)as gz:
  while block:=fi.read(1024*1024):rawsha.update(block);rawbytes+=len(block);gz.write(block)
check=hashlib.sha256()
with gzip.open(P/'scopes.jsonl.gz','rb')as f:
 while block:=f.read(1024*1024):check.update(block)
assert check.hexdigest()==rawsha.hexdigest();save(P/'payload.json',{'raw_bytes':rawbytes,'raw_sha256':rawsha.hexdigest(),'gzip_bytes':(P/'scopes.jsonl.gz').stat().st_size,'gzip_sha256':hashlib.file_digest((P/'scopes.jsonl.gz').open('rb'),'sha256').hexdigest()})
print('PP',pp,'armed',armed,'snapshots',snapcount,'all_cache_writes',len(allcache),'minfree',min(v for x in free.values()for v in x.values()),flush=True)
for x in summary:
 if x['source20_writes']:print('UB',x['pos'],x['tokens'],'kv_only',x['kv_only'],'cache writes',[(v['graph_index'],v['backend'],v['name'],v['origin_name'],v['value_owner'],v['dst_owner'],v['value_shape'],v['indices_name'])for v in x['source20_writes']],flush=True)

# Frozen-layout actual ON activation and mandatory KV-only completion.
assert allcache and len(allcache)==36
for u in summary:
 if not u['snapshot_count']:continue
 assert len(u['source20_writes'])==4,(u['pos'],len(u['source20_writes']))
 assert {x['cache_root_name']for x in u['source20_writes']}=={'cache_k_l20','cache_k_l23','cache_k_l30','cache_k_l38'}
 if u['kv_only']==20:assert all(x['name'].startswith('deferred_cache_update-')for x in u['source20_writes']if x['dst_owner']!='CUDA2')
# Verify first reader ordering for the captured full-tail prefix4096.
u=next(x for x in summary if x['pos']==4096);roots={x['root']:x for x in u['source20_writes']};readers={}
for line in F.open():
 row=json.loads(line)
 if row['id']!=u['scope']:continue
 graphs=sorted([x for x in row['calls']if x['kind']=='graph_compute'],key=lambda x:x['start'])
 for gi,g in enumerate(graphs):
  d=g.get('graph_snapshot')
  if not d:continue
  by={t['id']:t for t in d['tensors']}
  for ni,nid in enumerate(d['nodes']):
   t=by[nid]
   if t['op']in ['NONE','VIEW','RESHAPE','PERMUTE','TRANSPOSE','SET_ROWS']:continue
   for src in t['src']:
    if not src:continue
    root=by[src]['root']
    if root in roots and root not in readers:readers[root]={'graph':gi,'node':ni,'backend':g['backend'],'name':t['name'],'op':t['op']}
 break
assert len(readers)==4
for k,x in roots.items():assert(x['graph_index'],x['node_index'])<(readers[k]['graph'],readers[k]['node'])
assert all(str(P.parent/'candidate')in x for x in r['llama_maps'])
owner={'complete':True,'actual_origin_owners':sorted({x['origin_owner']for x in allcache}),'source20_replica_writes':36,'selected_target_chunks':9,'kv_only_chunks_flushed':sum(x['kv_only']==20 for x in summary if x['snapshot_count']),'same_source20_cache_roots':True,'write_before_all_four_first_readers':True,'private_llama_loaded':True,'readers':list(readers.values()),'graph_count_by_chunk':[(x['pos'],x['graph_count'])for x in summary],'min_free_mib':min(v for ph in free.values()for v in ph.values()),'scope':'Actual private ON module, Q3/Q8KV/DSparkK5/ctx92160; same-cohort repeated16token outputs/counters. This diagnostic observer is not native speed timing.'};save(P/'owner-checks.json',owner);print('ACTUAL_OWNER_ON_CONFIRMED',json.dumps(owner),flush=True)
