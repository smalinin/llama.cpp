from pathlib import Path
import sqlite3,json,collections,re,sys,bisect,functools

def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out

def length(xs):return sum(b-a for a,b in merge(xs))
def family(name):
 n=name.lower()
 if any(x in n for x in ['xfer_get','xfer_put','wait_flags','wait_flag','wait_seq']):return 'communication_and_wait'
 if any(x in n for x in ['attn','attention','indexer','lightning','cattn']):return 'attention_indexer'
 if 'strata::ds4::gguf::iq::' in n and any(x in n for x in ['(int)11','(int)12']):return 'Q3_Q4_matmul_or_dequant'
 if any(x in n for x in ['q3_k','q4_k','type)11','type)12','(ggml_type)11','(ggml_type)12']):return 'Q3_Q4_matmul_or_dequant'
 if any(x in n for x in ['gemm','gemv','mul_mat','mmq','mma','cutlass','_mv_kernel']):return 'other_matmul'
 if any(x in n for x in ['attn','attention','indexer','lightning','cattn']):return 'attention_indexer'
 if any(x in n for x in ['quant','dequant']):return 'activation_quant_or_dequant'
 if any(x in n for x in ['copy','concat','gather','rows','memcpy']):return 'pack_copy_gather'
 if any(x in n for x in ['topk','top_k','argmax','argsort','radix','mask','route']):return 'route_select_mask'
 if any(x in n for x in ['hc_','norm','rope','softmax','scale','silu','swiglu','sinkhorn']):return 'norm_HC_elementwise'
 return 'other'

def main():
 path=Path(sys.argv[1]);out=Path(sys.argv[2]);c=sqlite3.connect(path);c.text_factory=lambda b:b.decode('utf8','replace');tables={x[0]for x in c.execute("select name from sqlite_master where type='table'")};strings=dict(c.execute('select id,value from StringIds'));nvtx=[(a,b,tid,txt or strings.get(si,''))for a,b,tid,txt,si in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null')];phases=[(a,b,n)for a,b,t,n in nvtx if n.startswith('phase.')];assert len(phases)==3,phases
 rt=[]
 for tab in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if tab in tables:rt.extend((a,b,t,corr,strings.get(si,str(si)))for a,b,t,corr,si in c.execute('select start,end,globalTid,correlationId,nameId from '+tab))
 rt.sort();mapping={(t&~16777215,corr):(a,b,t,n)for a,b,t,corr,n in rt};bytid=collections.defaultdict(list)
 for a,b,t,n in nvtx:
  if not n.startswith('phase.'):bytid[t].append((a,b,n))
 for x in bytid.values():x.sort()
 @functools.lru_cache(maxsize=None)
 def role(api):
  if api is None:return 'unmapped'
  a,b,t,n=api;rs=[(x,y,z)for x,y,z in bytid[t]if x<=a and y>=b];labels=[z for x,y,z in sorted(rs,key=lambda r:r[0])]
  if any(z.startswith('decode.draft')for z in labels):return 'draft_seed'if any(z=='bridge.prefill'for z in labels)else'draft_forward'
  if any(z.startswith('decode.target')for z in labels):return 'target_forward'
  if 'proposal.generate'in labels:return'draft_proposal_outer'
  if any(z.startswith('bridge.')for z in labels):return'feature_bridge_outer'
  return 'no_role_nvtx'
 kernels=[(a,b,d,corr,pid,strings.get(si,str(si)))for a,b,d,corr,pid,si in c.execute('select start,end,deviceId,correlationId,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL')];copies=[];sets=[]
 if 'CUPTI_ACTIVITY_KIND_MEMCPY'in tables:copies=list(c.execute('select start,end,deviceId,correlationId,globalPid,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'))
 if 'CUPTI_ACTIVITY_KIND_MEMSET'in tables:sets=list(c.execute('select start,end,deviceId from CUPTI_ACTIVITY_KIND_MEMSET'))
 result={'sqlite':str(path),'diagnostics':list(c.execute('select severity,text from DIAGNOSTIC_EVENT where severity>=2'))if'DIAGNOSTIC_EVENT'in tables else[],'kernel_records':len(kernels),'memcpy_records':len(copies),'runtime_records':len(rt),'OSRT_available':'OSRT_API'in tables,'phases':[]}
 for lo,hi,phase in phases:
  dev=collections.defaultdict(list);kd=collections.defaultdict(list);productive=collections.defaultdict(list);api=collections.defaultdict(lambda:[0,0]);fams=collections.defaultdict(lambda:[0,0]);knames=collections.defaultdict(lambda:[0,0]);roles=collections.defaultdict(lambda:[0,0]);cp=collections.defaultdict(lambda:[0,0,0]);sync=[];corrmap=collections.Counter();kernelint=[]
  for a,b,d,corr,pid,n in kernels:
   if b<=lo or a>=hi:continue
   x,y=max(a,lo),min(b,hi);ms=(y-x)/1e6;dev[d].append((x,y));kd[d].append((x,y));kernelint.append((x,y));z=fams[family(n)];z[0]+=1;z[1]+=ms;z=knames[n];z[0]+=1;z[1]+=ms
   
   if family(n)!='communication_and_wait':productive[d].append((x,y))
   mapped=mapping.get((pid,corr));corrmap['mapped'if mapped else'unmapped']+=1;z=roles[role(mapped)];z[0]+=1;z[1]+=ms
  for a,b,d,corr,pid,nb,kind in copies:
   if b<=lo or a>=hi:continue
   x,y=max(a,lo),min(b,hi);dev[d].append((x,y));z=cp[str(kind)];z[0]+=1;z[1]+=nb;z[2]+=(y-x)/1e6
  for a,b,d in sets:
   if b>lo and a<hi:dev[d].append((max(a,lo),min(b,hi)))
  for a,b,t,corr,n in rt:
   if b<=lo or a>=hi:continue
   x,y=max(a,lo),min(b,hi);z=api[n];z[0]+=1;z[1]+=(y-x)/1e6
   if 'Synchronize'in n or 'StreamWait'in n:sync.append((x,y))
  produnion=length([i for ints in productive.values() for i in ints]);prodsum=sum(length(v) for v in productive.values())
  events=[]
  for d,ints in kd.items():
   for a,b in merge(ints):events.extend([(a,1),(b,-1)])
  events.sort();hist=collections.Counter();previous=lo;active=0
  for t,delta in events:
   hist[active]+=t-previous;previous=t;active+=delta
  hist[active]+=hi-previous
  busy=length([i for ints in dev.values()for i in ints]);kbusy=length(kernelint);sumdev=sum(length(x)for x in kd.values());wall=(hi-lo)/1e6
  toprows=lambda dd,keys:[dict(zip(keys,k if isinstance(k,tuple)else[k]),count=v[0],sum_ms=v[1])for k,v in sorted(dd.items(),key=lambda x:-x[1][1])]
  row={'phase':phase,'start':lo,'end':hi,'wall_ms':wall,'kernel_any_gpu_union_ms':kbusy/1e6,'any_activity_union_ms':busy/1e6,'observed_all_GPU_idle_pct':100*(1-busy/(hi-lo)),'mean_GPUs_with_active_kernel':sumdev/(hi-lo),'mean_active_GPU_count_when_kernel_running':sumdev/kbusy if kbusy else 0,'mean_GPUs_with_productive_kernel':prodsum/(hi-lo),'mean_productive_GPU_when_computing':prodsum/produnion if produnion else 0,'productive_kernel_union_ms':produnion/1e6,'per_device_productive_kernel_ms':{str(d):length(v)/1e6 for d,v in productive.items()},'GPU_kernel_concurrency_ms':{str(n):v/1e6 for n,v in sorted(hist.items())},'per_device_kernel_union_ms':{str(d):length(v)/1e6 for d,v in kd.items()},'kernel_families':toprows(fams,['family']),'top_kernels':toprows(knames,['kernel'])[:80],'kernel_roles':toprows(roles,['role']),'correlation_coverage':dict(corrmap),'copies':[{'kind':k,'count':v[0],'bytes':v[1],'sum_ms':v[2]}for k,v in cp.items()],'runtime_api':toprows(api,['api'])[:45],'sync_API_any_thread_union_ms':length(sync)/1e6}
  if'OSRT_API'in tables:row['OSRT_records']=c.execute('select count(*)from OSRT_API where end>? and start<?',(lo,hi)).fetchone()[0]
  result['phases'].append(row)
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items()if k not in ('phases','diagnostics')},indent=2));print('diagnostics',dict(collections.Counter(t for _,t in result['diagnostics'])))
 for x in result['phases']:print(x['phase'],'wall',round(x['wall_ms'],2),'meanGPU',round(x['mean_GPUs_with_active_kernel'],3),'idle%',round(x['observed_all_GPU_idle_pct'],2),'corr',x['correlation_coverage'],'kernelTop',[(r['family'],round(r['sum_ms'],1))for r in x['kernel_families'][:4]])
if __name__=='__main__':main()
