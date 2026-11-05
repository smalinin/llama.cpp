from pathlib import Path
import json,collections,statistics,subprocess,os
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts/shared-local'
if not(P/'scopes.jsonl').exists():print('No probe scopes yet.');raise SystemExit(0)
rows=[]
for l in(P/'scopes.jsonl').read_text().splitlines(True):
 if l.endswith('\n'):rows.append(json.loads(l))
byid={s['id']:s for s in rows};j=json.loads((P/'counts-results.json').read_text());out={'complete':j['complete'],'scopes':len(rows),'cases':[]}
def lineage(s):
 path=[s['kind']]
 while s['parent']in byid:s=byid[s['parent']];path.append(s['kind'])
 return '/'.join(reversed(path))
for case in sorted({s['case']for s in rows}):
 sr=[s for s in rows if s['case']==case];tot=collections.defaultdict(lambda:[0,0]);routes=collections.defaultdict(lambda:[0,0,0,0]);pc=collections.defaultdict(lambda:[0,0]);scopes=collections.defaultdict(lambda:[0,0]);slow=[]
 for s in sr:
  path=lineage(s);scopes[path][0]+=1;scopes[path][1]+=s['end']-s['start']
  for c in s['calls']:
   ns=c['end']-c['start']
   if c['depth']==0:
    k=(path,c['kind']);tot[k][0]+=1;tot[k][1]+=ns
   if c['kind']=='copy_iface':
    k=(path,c['source_backend'],c['backend'],c['src']['name'],c['async_accepted']);routes[k][0]+=1;routes[k][1]+=ns;routes[k][2]+=c['src']['bytes'];routes[k][3]=max(routes[k][3],ns)
   if c['kind']in ['event_sync','backend_sync','copy_iface','graph_compute']:
    k=(c['kind'],c['module'],c['offset']);pc[k][0]+=1;pc[k][1]+=ns
   if ns>10000000:slow.append({'scope':s['kind'],'pos':s['pos'],'tokens':s['tokens'],**c})
 summary={'case':case,'scope_ms':[{'path':k,'count':v[0],'ms':v[1]/1e6}for k,v in sorted(scopes.items(),key=lambda x:-x[1][1])],'top_level_host_calls':[{'scope':k[0],'kind':k[1],'count':v[0],'ms':v[1]/1e6}for k,v in sorted(tot.items(),key=lambda x:-x[1][1])],'copy_routes':[{'scope':k[0],'src_backend':k[1],'dst_backend':k[2],'tensor':k[3],'accepted':k[4],'count':v[0],'ms':v[1]/1e6,'bytes':v[2],'max_ms':v[3]/1e6}for k,v in sorted(routes.items(),key=lambda x:-x[1][1])],'callsites':[{'kind':k[0],'module':k[1],'offset':hex(k[2]),'count':v[0],'ms':v[1]/1e6}for k,v in sorted(pc.items(),key=lambda x:-x[1][1])],'slow_calls':sorted(slow,key=lambda c:c['end']-c['start'],reverse=True)[:12]}
 out['cases'].append(summary)
 if not j['complete']:summary['partial']=True
out['native_controls']=[{'prefix':n,'values':[(r['kind'],r['pp'],r['tg'])for r in j['requests']if r['prefix']==n]}for n in sorted({r['prefix']for r in j['requests']})]
(P/'host-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for c in out['cases']:
 print('CASE',c['case'],'scopes',[(x['path'],round(x['ms'],2))for x in c['scope_ms'][:5]])
 print('HOST',[(x['scope'],x['kind'],round(x['ms'],2))for x in c['top_level_host_calls'][:7]])
 print('ROUTES',[(x['scope'],x['src_backend'],x['dst_backend'],x['tensor'],round(x['ms'],2),x['count'])for x in c['copy_routes'][:5]])
print('complete',out['complete'])
