from pathlib import Path
import json,statistics,hashlib
P=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts')
def main():
 j=json.loads((P/'results.json').read_text());assert j['complete'] and j['source_files_unchanged'];rows=[];memory={};assert sum(len(a['requests']) for a in j['arms'])==52
 for a in j['arms']:
  assert a['exit_code']==0 and a['complete'] and not a['forced_kill'] and not a['warnings'] and not a['floor_breach'];assert not a['memory'].get('error')
  memory[a['label']]=min(v['min_free']/1048576 for phase in ['load','inference'] for v in a['memory'][phase].values() if isinstance(v,dict) and 'min_free' in v);assert memory[a['label']]>=512
  for n in [512,8192,98304]:
   q=[x for x in a['requests'] if x['prefix']==n and x['kind']=='measured'];assert len(q)==(2 if n==98304 else 3);acc=sum(x['draft_accepted'] for x in q);off=sum(x['draft_offered'] for x in q)
   r={'label':a['label'],'prefix':n,'repeats':len(q),'acceptance':acc/off if off else None,'draft_accepted':acc,'draft_offered':off,'unique_output_sequences':len({x['tokens_sha256'] for x in q})}
   for k in ['pp','tg','cpu_s','mean_cpu_cores','wall_s']:r[k]=statistics.median(x[k] for x in q);r[k+'_range']=[min(x[k] for x in q),max(x[k] for x in q)]
   rows.append(r)
 comparisons=[]
 for spec in ['off','on']:
  for n in [512,8192,98304]:
   own=next(r for r in rows if r['label']=='ours-'+spec and r['prefix']==n);maya=next(r for r in rows if r['label']=='maya-'+spec and r['prefix']==n)
   comparisons.append({'sidecar':spec,'prefix':n,**{k+'_maya_delta_pct':100*(maya[k]/own[k]-1) for k in ['pp','tg','cpu_s']}})
 out={'complete':True,'cohort':j['cohort'],'rows':rows,'comparisons':comparisons,'minimum_free_mib':memory,'protocol':j['protocol'],'historical_speed_comparisons_performed':False};(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
