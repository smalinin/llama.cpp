from pathlib import Path
import json,statistics
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts'
n=json.loads((P/'native-results.json').read_text());out={'native':[],'counter_cases':[]}
for prefix in [512,8192,98304]:
 arms={a['label']:[r for r in a['requests']if r['prefix']==prefix and r['kind']=='measured']for a in n['arms']}
 row={'prefix':prefix,'arms':{a:{m:statistics.mean(r[m]for r in rs)for m in ['pp','tg']}for a,rs in arms.items()}}
 for m in ['pp','tg']:
  off=statistics.mean(row['arms'][a][m]for a in ['off-pre','off-post']);row[m+'_off_mean']=off;row[m+'_gain_percent']=(row['arms']['on'][m]/off-1)*100
 out['native'].append(row)
records=[json.loads(x)for x in(P/'counts.jsonl').read_text().splitlines()]
for case in [4,8]:
 rs=[r for r in records if r['case']==case]
 waits={hex(off):{'calls':sum(w['calls']for r in rs for w in r['waits']if w['offset']==off),'seconds':sum(w['ns']for r in rs for w in r['waits']if w['offset']==off)/1e9}for off in sorted({w['offset']for r in rs for w in r['waits']})}
 out['counter_cases'].append({'case':case,'chunks':len(rs),'cpu_graphs':sum(len(r['cpu_graphs'])for r in rs),'cpu_seconds':sum(g['end']-g['start']for r in rs for g in r['cpu_graphs'])/1e9,'waits':waits})
out['native_memory_min_free_mib']={a['label']:{ph:min(v['min_free']for v in a['memory'][ph].values()if isinstance(v,dict))/2**20 for ph in ['load','inference']}for a in n['arms']}
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
