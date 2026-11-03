from pathlib import Path
import sqlite3,json,re,collections,bisect,statistics
P=Path(__file__).resolve().parent

def union(xs):
 total=0;end=-10**30
 for a,b in sorted(xs):
  if b>end:total+=b-max(a,end);end=b
 return total

def family(s):
 if re.search(r'mul_mat_(?:q|vec_q).*\(ggml_type\)(?:11|12)\b',s):return 'routed_Q3_Q4_matmul'
 if 'moe' in s.lower():return 'MoE_other'
 if 'lightning' in s.lower() or 'indexer' in s.lower():return 'indexer'
 if any(x in s.lower() for x in ['radix','argsort','top_k','topk','argmax','mask']):return 'sort_topk_mask_aux'
 if any(x in s.lower() for x in ['flash_attn','flash_attention','attn_ext','dsv41_attn','dsv4_attn']):return 'attention'
 if re.search(r'mul_mat_(?:q|vec_q)',s):return 'other_quant_matmul'
 if any(x in s.lower() for x in ['gemm','gemv','mul_mat','mma_','cutlass','ampere_','sm80_','sm89_']):return 'floating_matmul'
 if any(x in s.lower() for x in ['concat','cpy_','copy_','gather','get_rows']):return 'pack_copy_gather'
 if any(x in s.lower() for x in ['quantize','quant_q','quant_input']):return 'activation_quant'
 if any(x in s.lower() for x in ['hc_','hyper_connection','dsv4_hc']):return 'HC'
 return 'other_norm_elementwise_aux'

def attrs(s):
 d=dict(re.findall(r'(role|case|pos|tokens|bytes|flags|backend)=([^ ]+)',s))
 if s.startswith(('decode.','ubatch.')):d['role']=s.split()[0].split('.',1)[1]
 if s.startswith('bridge.'):d['bridge']=s.split()[0].split('.',1)[1]
 if s.startswith('checkpoint.'):
  d['role']=s.split()[0].split('.',1)[1]+('_prefill' if int(d.get('flags',0))&1 else '_generation');d['checkpoint']=True
 if s.startswith(('download ','upload ')):d['transfer']=s.split()[0];d['tensor']=s.split(' tensor=',1)[-1]
 if s.startswith('profiler.'):d['role']='drain'
 return d

def analyze_sql(path):
 c=sqlite3.connect(path);c.text_factory=lambda b:b.decode('utf8','replace');strings=dict(c.execute('select id,value from StringIds'))
 ranges=collections.defaultdict(list)
 for a,b,tid,s in c.execute('select start,end,globalTid,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS where end is not null'):
  if s and s.startswith(('decode.','ubatch.','bridge.','feature.','backend.','checkpoint.','download ','upload ','profiler.')):ranges[tid].append((a,b,s,attrs(s)))
 rt=[]
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if c.execute("select count(*) from sqlite_master where name=?",(table,)).fetchone()[0]:rt.extend((a,b,tid,corr,strings[n])for a,b,tid,corr,n in c.execute('select start,end,globalTid,correlationId,nameId from '+table))
 rt.sort();idx=collections.defaultdict(int);active=collections.defaultdict(list);mapping={};apis=collections.defaultdict(lambda:[0,0]);sync=collections.defaultdict(list)
 for r in ranges.values():r.sort(key=lambda x:(x[0],-x[1]))
 for a,b,tid,corr,name in rt:
  rr=ranges[tid];i=idx[tid];aa=[x for x in active[tid]if x[1]>=b]
  while i<len(rr)and rr[i][0]<=a:
   if rr[i][1]>=b:aa.append(rr[i])
   i+=1
  idx[tid]=i;active[tid]=aa;ctx={}
  for x in sorted(aa,key=lambda x:(x[0],-x[1])):ctx.update(x[3])
  mapping[(tid&~((1<<24)-1),corr)]=(ctx,name)
  role=ctx.get('role','unassigned');apis[(role,name)][0]+=1;apis[(role,name)][1]+=b-a
  if any(x in name.lower()for x in ['synchronize','eventrecord','streamwait']):sync[role].append((a,b))
 kernels=collections.defaultdict(lambda:[0,0]);ks=collections.defaultdict(lambda:[0,0]);ints=collections.defaultdict(list);devices=collections.defaultdict(list);unmapped=collections.Counter();posstats=collections.defaultdict(lambda:[0,0]);counters=collections.Counter()
 for a,b,dev,corr,pid,n in c.execute('select start,end,deviceId,correlationId,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
  ctx,api=mapping.get((pid,corr),({},''));role=ctx.get('role','unassigned');name=strings[n];fam=family(name);ms=(b-a)/1e6
  counters['kernels']+=1
  if not api:unmapped['kernels']+=1
  kernels[(role,fam)][0]+=1;kernels[(role,fam)][1]+=ms;ks[(role,name)][0]+=1;ks[(role,name)][1]+=ms
  ints[role].append((a,b));devices[(role,dev)].append((a,b))
  pos=int(ctx.get('pos',-1));bucket='early0_8K'if 0<=pos<8192 else 'late90K'if pos>=90112 else 'middle';posstats[(role,bucket,fam)][0]+=1;posstats[(role,bucket,fam)][1]+=ms
 copies=collections.defaultdict(lambda:[0,0,0]);transfers=collections.defaultdict(lambda:[0,0,0]);copyapis=collections.defaultdict(lambda:[0,0,0]);ci=collections.defaultdict(list)
 for a,b,dev,corr,pid,nb,ck in c.execute('select start,end,deviceId,correlationId,globalPid,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
  ctx,api=mapping.get((pid,corr),({},''));role=ctx.get('role','unassigned');counters['copies']+=1
  if not api:unmapped['copies']+=1
  key=(role,str(ck));v=copies[key];v[0]+=1;v[1]+=nb;v[2]+=(b-a)/1e6
  v=copyapis[(role,api,str(ck))];v[0]+=1;v[1]+=nb;v[2]+=(b-a)/1e6
  name=ctx.get('tensor','unlabeled');v=transfers[(role,str(ck),name)];v[0]+=1;v[1]+=nb;v[2]+=(b-a)/1e6
  ci[role].append((a,b));devices[(role,dev)].append((a,b))
 def rows(d,keys,vals):return [dict(zip(keys,k),**dict(zip(vals,v)))for k,v in sorted(d.items(),key=lambda x:-x[1][-1])]
 return {'sqlite':str(path),'counts':dict(counters),'unmapped':dict(unmapped),'kernel_families':rows(kernels,['role','family'],['count','sum_ms']),'kernels':rows(ks,['role','kernel'],['count','sum_ms']),'position_families':rows(posstats,['role','bin','family'],['count','sum_ms']),'copy_apis':rows(copyapis,['role','api','copy_kind'],['count','bytes','sum_ms']),'copies':rows(copies,['role','copy_kind'],['count','bytes','sum_ms']),'transfers':rows(transfers,['role','copy_kind','tensor'],['count','bytes','sum_ms']),'api':rows(apis,['role','api'],['count','sum_ns']),'sync_api_union_ms':{k:union(v)/1e6 for k,v in sync.items()},'kernel_union_ms':{k:union(v)/1e6 for k,v in ints.items()},'copy_union_ms':{k:union(v)/1e6 for k,v in ci.items()},'gpu_union_ms':{k:union(ints[k]+ci[k])/1e6 for k in ints.keys()|ci.keys()},'per_device_gpu_union_ms':[{'role':r,'device':d,'ms':union(v)/1e6}for(r,d),v in devices.items()]}

def cpu_summary(events,case):
 es=[e for e in events if e.get('case')==case];out={}
 for role in sorted(set(e['role']for e in es)):
  rr=[e for e in es if e['role']==role];stats={}
  for ev in sorted(set(e['event']for e in rr)):
   rows=[e for e in rr if e['event']==ev];dur=[e['end_ns']-e['begin_ns']for e in rows];stats[ev]={'count':len(rows),'sum_ms':sum(dur)/1e6,'union_ms':union([(e['begin_ns'],e['end_ns'])for e in rows])/1e6,'bytes':sum(e.get('bytes',0)for e in rows)}
  out[role]=stats
 bridge=[e for e in es if e['event']=='bridge'and e['role']=='target_prefill'];exclusive=0
 for e in bridge:
  a,b=e['begin_ns'],e['end_ns'];child=[(max(a,x['begin_ns']),min(b,x['end_ns']))for x in es if x['event']in['decode','feature_get','feature_get_packed']and x['begin_ns']>=a and x['end_ns']<=b];exclusive+=b-a-union(child)
 out['bridge_prefill_exclusive_ms']=exclusive/1e6
 groups=collections.defaultdict(list)
 for e in es:
  if e['event']=='cpu_graph'and e['role']=='target_prefill':groups[(e['pos'],e['tokens'])].append(e)
 out['target_cpu_graphs_per_ubatch']=dict(collections.Counter(len(v)for v in groups.values()));out['target_cpu_ordered_ms']=[sum(sorted(v,key=lambda e:e['begin_ns'])[i]['end_ns']-sorted(v,key=lambda e:e['begin_ns'])[i]['begin_ns']for v in groups.values())/1e6 for i in range(3)]if all(len(v)==3 for v in groups.values())else None
 return out

if __name__=='__main__':
 report=json.loads((P/'results.json').read_text());summary={'variants':[]}
 for variant in report['variants']:
  if variant['variant']=='native':continue
  root=P/variant['variant'];events=[json.loads(x)for x in(root/'events.jsonl').read_text().splitlines()];v={'variant':variant['variant'],'complete':variant['complete'],'cases':[],'cpu_metadata':[e for e in events if e['event']=='cpu_metadata']}
  for request in variant['requests']:
   x={'id':request['id'],'n':request['n'],'kind':request['kind'],'timings':request['response']['timings'],'cpu':cpu_summary(events,request['id'])}
   if request['trace']:
    seq={2:1,5:2,9:3}[request['id']];path=root/f'capture.{seq}.sqlite'
    if path.exists():print('Analyze',path,flush=True);x['gpu']=analyze_sql(path)
   v['cases'].append(x)
  summary['variants'].append(v)
 (P/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
 for v in summary['variants']:
  print('\nVARIANT',v['variant'])
  for x in v['cases']:
   if not x.get('gpu'):continue
   print(x['n'],x['timings']['prompt_ms'],'CPU',json.dumps(x['cpu']));print('GPU',json.dumps(x['gpu']['kernel_families']));print('COPIES',json.dumps(x['gpu']['copies']));print('UNIONS',x['gpu']['gpu_union_ms'],'UNMAPPED',x['gpu']['unmapped'])
