from pathlib import Path
import sqlite3,collections,json,re
from analyze import attrs
P=Path(__file__).resolve().parent

def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def total(xs):return sum(b-a for a,b in xs)
def overlap(a,b):
 i=j=n=0
 while i<len(a) and j<len(b):
  n+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return n
out=[]
for seq in [2,3]:
 c=sqlite3.connect(P/f'sidecar/capture.{seq}.sqlite');c.text_factory=lambda b:b.decode('utf8','replace');c.row_factory=sqlite3.Row;strings={r['id']:r['value']for r in c.execute('select id,value from StringIds')};ranges=collections.defaultdict(list);nv=[]
 for r in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end>start'):
  label=r['text']or strings.get(r['textId'],'');d=attrs(label)
  if label.startswith('backend.CUDA'):
   m=re.match(r'backend.CUDA(\d+)',label)
   if m:d['backend_device']=int(m[1])
  if label.startswith(('decode.','ubatch.','bridge.','feature.','backend.','checkpoint.','download ','upload ','profiler.','pack.','copy_')):
   item=(r['start'],r['end'],label,d);ranges[r['globalTid']].append(item);nv.append(item)
 for rr in ranges.values():rr.sort(key=lambda x:(x[0],-x[1]))
 rt=[]
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if c.execute('select count(*) from sqlite_master where name=?',(table,)).fetchone()[0]:
   rt.extend((r['start'],r['end'],r['globalTid'],r['correlationId'],strings.get(r['nameId'],''),table)for r in c.execute('select * from '+table))
 active=collections.defaultdict(list);cursor=collections.defaultdict(int);mapping={};api=[]
 for a,b,tid,corr,name,table in sorted(rt):
  rr=ranges[tid];i=cursor[tid];aa=[z for z in active[tid]if z[1]>=b]
  while i<len(rr)and rr[i][0]<=a:
   if rr[i][1]>=b:aa.append(rr[i])
   i+=1
  cursor[tid]=i;active[tid]=aa;ctx={}
  for z in sorted(aa,key=lambda z:(z[0],-z[1])):ctx.update(z[3])
  mapping[(tid&~((1<<24)-1),corr)]=ctx
  if table.endswith('RUNTIME'):api.append((a,b,name,ctx.get('role','unassigned')))
 activities=[];backend_pairs=collections.Counter()
 for table in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
  if not c.execute('select count(*) from sqlite_master where name=?',(table,)).fetchone()[0]:continue
  for r in c.execute('select * from '+table+' where end>start'):
   activities.append((r['start'],r['end']));ctx=mapping.get((r['globalPid'],r['correlationId']),{})
   if table.endswith('KERNEL')and'backend_device'in ctx:backend_pairs[(ctx['backend_device'],r['deviceId'])]+=1
 spanrows=[z for z in nv if z[2].startswith(('decode.target_prefill','bridge.prefill','checkpoint.target'))and z[3].get('role')not in ['target_generation']]
 begin=min(z[0]for z in spanrows);end=max(z[1]for z in spanrows);gpu=merge((max(a,begin),min(b,end))for a,b in activities if a<end and b>begin);gap=[];prev=begin
 for a,b in gpu:
  if a>prev:gap.append([prev,a])
  prev=max(prev,b)
 if prev<end:gap.append([prev,end])
 apis=collections.defaultdict(list)
 for a,b,name,role in api:
  if a<end and b>begin:apis[name].append((max(begin,a),min(end,b)))
 apimerge={k:merge(v)for k,v in apis.items()};api_union=merge(x for v in apimerge.values()for x in v)
 cpu=merge((max(a,begin),min(b,end))for a,b,label,d in nv if label.startswith('backend.CPU')and a<end and b>begin)
 info=[dict(r)for r in c.execute('select id,name,busLocation,uuid,cuDevice from TARGET_INFO_GPU')]
 row={'capture':seq,'n':8192 if seq==2 else 98304,'prompt_span_ns':[begin,end],'prompt_span_ms':(end-begin)/1e6,'gpu_any_activity_ms':total(gpu)/1e6,'no_recorded_gpu_activity_ms':total(gap)/1e6,'gap_inside_runtime_api_ms':overlap(gap,api_union)/1e6,'gap_inside_CPU_graph_ms':overlap(gap,cpu)/1e6,'api_gap_rows':sorted([{'api':k,'api_union_ms':total(v)/1e6,'no_gpu_overlap_ms':overlap(gap,v)/1e6}for k,v in apimerge.items()],key=lambda x:-x['no_gpu_overlap_ms'])[:15],'backend_cuda_to_cupti_device_kernel_counts':[{'backend_cuda':k[0],'cupti_device':k[1],'count':v}for k,v in sorted(backend_pairs.items())],'nsys_gpu_info':info};out.append(row)
 print('N',row['n'],'span',round(row['prompt_span_ms'],1),'GPUidle',round(row['no_recorded_gpu_activity_ms'],1),'APIgap',round(row['gap_inside_runtime_api_ms'],1),'CPUGraphgap',round(row['gap_inside_CPU_graph_ms'],1),'top',row['api_gap_rows'][:3],flush=True)
(P/'gaps.json').write_text(json.dumps(out,indent=2)+'\n')
