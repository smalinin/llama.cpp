from pathlib import Path
import sqlite3,json,bisect,collections,hashlib,re
P=Path(__file__).parent
path=Path('/tmp/dsv41-m2-residual-profile-20261103/sidecar/capture.2.sqlite')
c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
c.text_factory=lambda b:b.decode('utf8','replace')
strings=dict(c.execute('select id,value from StringIds'))
ranges=collections.defaultdict(list)
backends=collections.defaultdict(list)
for a,b,t,s in c.execute('select start,end,globalTid,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS where end is not null'):
 if not s:continue
 if s.startswith('ubatch.target_prefill'):ranges[t].append((a,b,s))
 if s.startswith('backend.'):backends[t].append((a,b,s))
for v in backends.values():v.sort()
copies=collections.defaultdict(list)
for row in c.execute('select start,end,deviceId,correlationId,globalPid,bytes,copyKind,srcDeviceId,dstDeviceId,contextId,srcContextId,dstContextId from CUPTI_ACTIVITY_KIND_MEMCPY'):
 a,b,dev,corr,pid,n,kind,src,dst,ctx,sc,dc=row
 copies[pid,corr].append({'start':a,'end':b,'device':dev,'bytes':n,'kind':kind,'source_device':src,'destination_device':dst,'context':ctx,'source_context':sc,'destination_context':dc})
matched=[]
for a,b,t,corr,nid in c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
 name=strings[nid]
 if not name.startswith('cudaMemcpyPeerAsync'):continue
 ub=[r for r in ranges[t]if r[0]<=a and b<=r[1]]
 if not ub:continue
 phys=copies.get((t&~((1<<24)-1),corr),[])
 if not phys or max(x['bytes']for x in phys)not in [41943040,41615360]:continue
 bb=backends[t];index=bisect.bisect_left(bb,(b,))
 nextb=bb[index]if index<len(bb)else None
 if not nextb or not nextb[2].startswith(('backend.CUDA1 ','backend.CUDA2 ')):continue
 matched.append({'start':a,'end':b,'api':name,'thread':t,'correlation':corr,'ubatch':ub[-1][2],'next_backend':nextb,'copy_records':phys})
groups=collections.defaultdict(list)
for m in matched:groups[(m['thread'],m['next_backend'][0])].append(m)
outgroups=[]
for k,v in groups.items():
 outgroups.append({'backend':v[0]['next_backend'][2],'ubatch':v[0]['ubatch'],'api_calls':len(v),'api_host_sum_ms':sum(x['end']-x['start']for x in v)/1e6,'copy_records':sum(len(x['copy_records'])for x in v),'physical_byte_sum':sum(y['bytes']for x in v for y in x['copy_records']),'copy_engine_sum_ms':sum(y['end']-y['start']for x in v for y in x['copy_records'])/1e6})
by=collections.defaultdict(lambda:collections.Counter())
for g in outgroups:
 name=g['backend'].split()[0]
 d=by[name];d['groups']+=1;d['api_calls']+=g['api_calls'];d['api_host_sum_ms']+=g['api_host_sum_ms'];d['copy_records']+=g['copy_records'];d['physical_byte_sum']+=g['physical_byte_sum'];d['copy_engine_sum_ms']+=g['copy_engine_sum_ms']
report={'complete':True,'scope':'Historical M2 ctx1003528K capture,CPU-only read-only reanalysis. Selection by full HC-sized memcpy record and immediately following CUDA1/2 backend graph on same thread.Not pointer identity proof;compare with independently captured alias audit. No new GPU measurement or speed A/B.','sqlite_locator':str(path),'sqlite_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'groups':outgroups,'summary':dict(by),'calls':matched}
(P/'profile-costs.json').write_text(json.dumps(report,indent=2)+'\n')
print('SUMMARY',json.dumps(dict(by),indent=2))
print('first',json.dumps(matched[:1],indent=2))
print('groups',collections.Counter(g['api_calls']for g in outgroups))
