from pathlib import Path
import sqlite3,json,re,collections,bisect,hashlib
P=Path(__file__).parent

def analyze(path,case,events):
 nodes=[e for e in events if e.get('case')==case and e['event']=='ix_node'];expect=collections.defaultdict(set);details={}
 for e in nodes:
  for s in e['src']:
   if not re.match(r'CUDA\d+#(?:idx_score-20|candidate_mask-20)#',s['name']):continue
   cat='score20'if'#idx_score-20#'in s['name']else'candidate_mask20';key=(e['pos'],e['backend'],s['bytes']);expect[key].add(cat);details[key,cat]={'source_copy':s,'consumer':e['dst']}
 c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.text_factory=lambda b:b.decode('utf8','replace');strings=dict(c.execute('select id,value from StringIds'));back=collections.defaultdict(list);ub=collections.defaultdict(list)
 for a,b,t,s in c.execute('select start,end,globalTid,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS where end is not null'):
  if not s:continue
  if s.startswith('backend.'):back[t].append((a,b,s))
  elif s.startswith('ubatch.target_prefill'):ub[t].append((a,b,s))
 for v in back.values():v.sort()
 phys=collections.defaultdict(list)
 for a,b,d,co,pid,n,k,sd,dd in c.execute('select start,end,deviceId,correlationId,globalPid,bytes,copyKind,srcDeviceId,dstDeviceId from CUPTI_ACTIVITY_KIND_MEMCPY'):phys[pid,co].append(dict(start=a,end=b,device=d,bytes=n,kind=k,src=sd,dst=dd))
 matches=[];excluded=[]
 for a,b,t,co,nid in c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
  name=strings[nid]
  if not name.startswith('cudaMemcpyPeerAsync'):continue
  ur=[r for r in ub[t]if r[0]<=a and b<=r[1]]
  if not ur:continue
  p=phys.get((t&~((1<<24)-1),co),[])
  if not p or len({r['bytes']for r in p})!=1:continue
  nb=p[0]['bytes'];bb=back[t];i=bisect.bisect_left(bb,(b,));nxt=bb[i]if i<len(bb)else None
  if not nxt or nxt[0]>ur[-1][1]:continue
  bd=nxt[2].split()[0][8:];pos=int(re.search(r'pos=(-?\d+)',nxt[2])[1]);key=pos,bd,nb;cats=expect.get(key,set())
  if len(cats)!=1:continue
  cat=next(iter(cats));m={'category':cat,'pos':pos,'logical_bytes':nb,'api_start':a,'api_end':b,'api':name,'correlation':co,'consumer_backend_range':nxt,'metadata':details[key,cat],'physical':p}
  # CUPTI src/dst ordinal fields may be null. Device IDs on D2H/H2D legs constrain the staged path.
  dest=int(bd[4:]);m['destination_metadata_ordinal']=dest
  assert any(r['kind']==1 and r['device']==dest for r in p),m
  assert any(r['kind']==2 and r['device']==(2 if cat=='score20'else 3)for r in p),m
  matches.append(m)
 groups=collections.defaultdict(lambda:collections.Counter())
 for m in matches:
  v=groups[m['category']];v['calls']+=1;v['logical_bytes']+=m['logical_bytes'];v['api_host_ns']+=m['api_end']-m['api_start'];v['physical_records']+=len(m['physical']);v['physical_bytes']+=sum(p['bytes']for p in m['physical']);v['physical_sum_ns']+=sum(p['end']-p['start']for p in m['physical'])
 return {'locator':str(path),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'case':case,'tool_diagnostics':[dict(timestamp=a,severity=sev,text=txt)for a,sev,txt in c.execute('select timestamp,severity,text from DIAGNOSTIC_EVENT')],'scope':'Metadata-constrained full-size copy/API correlation: source clone name/shape/backend in the next same-thread destination graph and containing prefill ubatch, then exact CUPTI correlation ID. No CUDA source/destination pointers were intercepted; do not describe this as a pointer-level transfer proof. Distinct expected categories required; unmatched/ambiguous APIs excluded. Sums are not critical-path/wall-time savings.','metadata_expectation_count':len(expect),'summary':dict(groups),'matches':matches}

if __name__=='__main__':
 es=[json.loads(l)for l in(P/'sidecar/events.jsonl').open()];r=json.load(open(P/'profile-results.json'));out={'complete':False,'cases':[]}
 for seq,req in enumerate([q for q in r['variants'][0]['requests']if q['trace']],1):
  x=analyze(P/f'sidecar/capture.{seq}.sqlite',req['id'],es);x['prompt_n']=req['n'];out['cases'].append(x);print(req['n'],x['metadata_expectation_count'],x['summary'])
 out['complete']=r['complete'];(P/'copy-correlation.json').write_text(json.dumps(out,indent=2)+'\n')
