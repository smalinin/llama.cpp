from pathlib import Path
import json,collections,subprocess
R=Path('/home/sergei/Github/llama.cpp');D=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts/shared-local'
p=D/'scopes.jsonl'
if not p.exists():print('No API scopes yet.');raise SystemExit(0)
rows=[json.loads(x)for x in p.read_text().splitlines(True)if x.endswith('\n')];ids={r['id']:r for r in rows};agg=collections.defaultdict(lambda:[0,0,0,0,0]);pcs=collections.defaultdict(lambda:[0,0,0,0,0]);g_ms=collections.defaultdict(lambda:[0,0]);moe=collections.defaultdict(int)
def role(r):
 while r['parent']in ids:r=ids[r['parent']]
 return r['kind']
for r in rows:
 for a in r.get('cuda_apis',[]):
  if a['depth']!=0:continue
  k=(role(r),a['kind']);v=agg[k];v[0]+=a['count'];v[1]+=a['ns'];v[2]=max(v[2],a['max_ns']);v[3]+=a['bytes'];v[4]+=a['errors']
  k=(role(r),a['kind'],a['module'],a['offset']);v=pcs[k];v[0]+=a['count'];v[1]+=a['ns'];v[2]=max(v[2],a['max_ns']);v[3]+=a['bytes'];v[4]+=a['errors']
 for c in r['calls']:
  if c['kind']=='graph_compute':
   k=(role(r),c['backend']);g_ms[k][0]+=1;g_ms[k][1]+=c['end']-c['start']
   for m in c.get('moe',[]):moe[(c['backend'],m['type'],m['weights'],tuple(m['w_ne']),tuple(m['x_ne']))]+=1
j={'complete':json.loads((D/'counts-results.json').read_text())['complete'],'scope_count':len(rows),'top_apis':[{'role':k[0],'kind':k[1],'count':v[0],'ms':v[1]/1e6,'max_ms':v[2]/1e6,'bytes':v[3],'errors':v[4]}for k,v in sorted(agg.items(),key=lambda x:-x[1][1])],'callsites':[{'role':k[0],'kind':k[1],'module':k[2],'offset':hex(k[3]),'count':v[0],'ms':v[1]/1e6,'max_ms':v[2]/1e6,'bytes':v[3],'errors':v[4]}for k,v in sorted(pcs.items(),key=lambda x:-x[1][1])],'graph_host_ms':[{'role':k[0],'backend':k[1],'count':v[0],'ms':v[1]/1e6}for k,v in g_ms.items()],'moe_shapes':[{'backend':k[0],'type':k[1],'weights':k[2],'w_ne':k[3],'x_ne':k[4],'count':v}for k,v in moe.items()]}
(D/'summary.json').write_text(json.dumps(j,indent=2)+'\n');print('scope_count',len(rows),'complete',j['complete']);print('TOP',[(x['role'],x['kind'],round(x['ms'],2),x['count'])for x in j['top_apis'][:12]]);print('CALLERS',[(x['kind'],x['offset'],round(x['ms'],2),x['count'])for x in j['callsites'][:8]])
