from pathlib import Path
import collections,json,sqlite3,sys,re
p=Path(sys.argv[1]);out=Path(sys.argv[2]);c=sqlite3.connect(p);c.text_factory=lambda b:b.decode("utf8","replace");s=dict(c.execute('select id,value from StringIds'));rows=[(a,b,t,n or s.get(i,''))for a,b,t,n,i in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null')];ph=[x for x in rows if x[3].startswith('phase.')];result=[]
for lo,hi,tid,name in ph:
 hist=collections.defaultdict(lambda:[0,0.,[]]);scopes=[]
 for a,b,t,n in rows:
  if n.startswith('phase.')or b<=lo or a>=hi:continue
  short=re.sub(r'pos=-?\d+ ','',n);k=re.sub(r'phase=\d+ ','',short);ms=(min(b,hi)-max(a,lo))/1e6;v=hist[k];v[0]+=1;v[1]+=ms;v[2].append(ms)
  if n.startswith(('decode.','ubatch.','proposal.','bridge.')):scopes.append({'start':a,'end':b,'tid':t,'scope':n})
 for v in hist.values():v[2].sort()
 result.append({'phase':name,'wall_ms':(hi-lo)/1e6,'scopes_summary':[{'scope':k,'count':v[0],'sum_CPU_scope_ms':v[1],'median_CPU_scope_ms':v[2][len(v[2])//2]}for k,v in hist.items()],'scopes':scopes})
report={'sqlite':str(p),'phases':result,'scope':'Nested NVTX host scope durations include enqueue and synchronization; sums across nesting are not additive GPU time.'};out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items()if k!='scopes'}for x in result],indent=2))
