from pathlib import Path
import json,sqlite3,collections,re,hashlib
P=Path(__file__).parent

def union(xs):
 end=-10**30;out=0
 for a,b in sorted(xs):
  if b>end:out+=b-max(a,end);end=b
 return out

def category(s):
 if 'idx_score-20' in s:return 'score20'
 if 'candidate_mask-20' in s:return 'candidate_mask20'
 if 'candidate_pin' in s:return 'candidate_pin'
 if 'candidate' in s:return 'candidate_other'
 return None

def extract(path):
 c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.text_factory=lambda b:b.decode('utf8','replace');strings=dict(c.execute('select id,value from StringIds'));ranges=collections.defaultdict(list)
 for a,b,t,s in c.execute('select start,end,globalTid,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS where end is not null'):
  if s and s.startswith(('ubatch.','decode.','backend.','copy_async ','copy_sync ','upload ','upload.sync ','ix.op ','profiler.')):ranges[t].append((a,b,s))
 for r in ranges.values():r.sort(key=lambda x:(x[0],-x[1]))
 idx=collections.defaultdict(int);active=collections.defaultdict(list);mapping={};logical=collections.defaultdict(lambda:collections.Counter());apis=collections.defaultdict(lambda:collections.Counter());opapis=collections.defaultdict(lambda:collections.Counter());copyrows=[];krows=[]
 rt=[]
 for tab in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if c.execute('select count(*) from sqlite_master where name=?',(tab,)).fetchone()[0]:rt.extend((a,b,t,corr,strings[n])for a,b,t,corr,n in c.execute('select start,end,globalTid,correlationId,nameId from '+tab))
 rt.sort()
 for a,b,t,corr,name in rt:
  rr=ranges[t];i=idx[t];aa=[x for x in active[t]if x[1]>=b]
  while i<len(rr)and rr[i][0]<=a:
   if rr[i][1]>=b:aa.append(rr[i])
   i+=1
  idx[t]=i;active[t]=aa
  ub=next((x for x in reversed(aa)if x[2].startswith('ubatch.')),None);role=ub[2].split()[0][7:]if ub else 'other'
  cp=next((x for x in reversed(aa)if x[2].startswith(('copy_async ','copy_sync ','upload ','upload.sync '))),None)
  op=next((x for x in reversed(aa)if x[2].startswith('ix.op ')),None)
  if role!='target_prefill':continue
  if cp and category(cp[2]) and 'Memcpy' in name:
   cat=category(cp[2]);v=apis[cat,name];v['count']+=1;v['sum_ns']+=b-a
   pos=int(re.search(r'pos=(-?\d+)',ub[2])[1]);mapping[t&~((1<<24)-1),corr]={'category':cat,'copy_range':cp[2],'pos':pos,'api':name}
  elif op:
   label=op[2];v=opapis[label.split(' pos=')[0],label.split(' op=')[-1]];v['count']+=1;v['sum_ns']+=b-a
   mapping[t&~((1<<24)-1),corr]={'op_range':label,'api':name}
 for t,rr in ranges.items():
  ubs=[x for x in rr if x[2].startswith('ubatch.target_prefill')]
  # Logical requests count only the outer async tensor-copy range; fallback nested sync is not another request.
  for a,b,s in rr:
   if not s.startswith('copy_async ')or not category(s)or 'role=target_prefill 'not in s:continue
   cat=category(s);v=logical[cat];v['count']+=1;v['bytes']+=int(re.search(r'bytes=(\d+)',s)[1]);v['host_ns']+=b-a
 for row in c.execute('select start,end,deviceId,correlationId,globalPid,bytes,copyKind,srcDeviceId,dstDeviceId from CUPTI_ACTIVITY_KIND_MEMCPY'):
  a,b,d,corr,pid,n,k,sd,dd=row;m=mapping.get((pid,corr))
  if m and m.get('category'):copyrows.append(dict(start=a,end=b,device=d,bytes=n,kind=k,src=sd,dst=dd,correlation=corr,**m))
 for row in c.execute('select start,end,deviceId,correlationId,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
  a,b,d,corr,pid,n=row;m=mapping.get((pid,corr))
  if m and m.get('op_range'):krows.append(dict(start=a,end=b,device=d,kernel=strings[n],correlation=corr,**m))
 physical=collections.defaultdict(lambda:collections.Counter());kernels=collections.defaultdict(lambda:collections.Counter())
 for r in copyrows:
  v=physical[r['category'],r['kind'],r['src'],r['dst']];v['count']+=1;v['bytes']+=r['bytes'];v['sum_ns']+=r['end']-r['start']
 for r in krows:
  label=r['op_range'].split(' op=')[-1];v=kernels[label,r['device'],r['kernel']];v['count']+=1;v['sum_ns']+=r['end']-r['start']
 def rows(d,keys):return[dict(zip(keys,k if isinstance(k,tuple)else[k]),**v)for k,v in sorted(d.items())]
 return {'locator':str(path),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'logical':dict(logical),'copy_apis':rows(apis,['category','api']),'physical':rows(physical,['category','kind','src','dst']),'physical_union_ns':{cat:union([(r['start'],r['end'])for r in copyrows if r['category']==cat])for cat in logical},'kernels':rows(kernels,['op_label','device','kernel']),'op_api':rows(opapis,['prefix','op_label']),'copy_records':copyrows,'op_kernel_records':krows}

if __name__=='__main__':
 report=json.load(open(P/'profile-results.json'));es=[json.loads(l)for l in(P/'sidecar/events.jsonl').open()];out={'complete':False,'scope':'Diagnostic only. Logical copy requests, CUPTI physical legs and host API/kernel sums are separate; no PP gain inferred from their sum. Same-thread NVTX+API correlation binds selected copies and operators to tensor names. Non-interposed/fused kernels are not charged to a selected op.','cases':[]}
 for seq,req in enumerate([r for r in report['variants'][0]['requests']if r['trace']],1):
  case=req['id'];nodes=[e for e in es if e.get('case')==case and e['event']=='ix_node'];copies=[e for e in es if e.get('case')==case and e['event']=='ix_copy_metadata'];gpu=extract(P/f'sidecar/capture.{seq}.sqlite')
  selected=[]
  for e in nodes:
   if e['dst']['name']in['idx_score-20','candidate_block_score-20','candidate_keep-20','candidate_mask-20'] or any(s['name']=='candidate_mask-20'or s['name'].endswith('#candidate_mask-20#0')for s in e['src']):selected.append(e)
  x={'id':case,'prompt_n':req['n'],'timings':req['response']['timings'],'target_ubatches':req['target_ubatches'],'kv_only_ubatches':req['kv_only_ubatches'],'draft_seed_rows':req['draft_seed_rows'],'metadata_node_count':len(nodes),'copy_metadata_count':len(copies),'selected_nodes':selected,'copy_metadata':copies,'gpu':gpu};out['cases'].append(x)
 out['complete']=report['complete'];(P/'indexer-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 for x in out['cases']:
  print(x['prompt_n'],'ubatches',x['target_ubatches'],'KVonly',x['kv_only_ubatches'],'logical',json.dumps(x['gpu']['logical']),'physical',json.dumps(x['gpu']['physical']),flush=True)
