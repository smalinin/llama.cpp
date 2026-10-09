from pathlib import Path
import json,hashlib,statistics,subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/sergei/Github/llama.cpp')
def read(p):return json.loads(Path(p).read_text())
def first(x,y):return next((i for i,(a,b) in enumerate(zip(x,y)) if a!=b),None)
labels=['baseline','svg','python','explain','ratio2-long-fresh','ratio2-long-cache'];result={'before':[],'final':[],'final_requests':0}
for label in labels:
 off=read(r/'before-off'/f'{label}-response.json');on=read(r/'before-n3'/f'{label}-response.json');result['before'].append({'case':label,'equal':off['tokens']==on['tokens'],'first_difference':first(off['tokens'],on['tokens']),'off_tps':off['timings']['predicted_per_second'],'n3_tps':on['timings']['predicted_per_second']})
for label in labels:
 row={'case':label};cases=[label] if label.startswith('ratio2-') else [label,label+'-repeat2']
 for config in ['off','n1','n3']:
  tps=[];accepted=drafted=0
  for case in cases:
   x=read(r/('final-'+config)/(case+'-response.json'));ref=read(r/'final-off'/(case+'-response.json'))
   assert x['tokens']==ref['tokens'] and x['content']==ref['content'],(config,case,first(x['tokens'],ref['tokens']))
   assert len(x['tokens'])==len(ref['tokens'])== (1024 if label.startswith('ratio2-') else 256)
   if case.endswith('cache'):assert x['timings']['cache_n']==6613
   if config!='off':assert x['timings']['draft_n']>0
   tps.append(x['timings']['predicted_per_second']);accepted+=x['timings'].get('draft_n_accepted',0);drafted+=x['timings'].get('draft_n',0);result['final_requests']+=1
  row[config]={'tokens':len(x['tokens']),'tps':tps,'median_tps':statistics.median(tps),'drafted':drafted,'accepted':accepted,'acceptance':accepted/drafted if drafted else None,'warm_tps':tps[-1]}
  if config!='off':
   row[config]['change_vs_off_pct']=100*(row[config]['median_tps']/row['off']['median_tps']-1)
   row[config]['warm_change_vs_off_pct']=100*(row[config]['warm_tps']/row['off']['warm_tps']-1)
 result['final'].append(row)
for config in ['off','n1','n3']:
 fresh=read(r/('final-'+config)/'ratio2-long-fresh-response.json');cached=read(r/('final-'+config)/'ratio2-long-cache-response.json');assert fresh['tokens']==cached['tokens'] and fresh['content']==cached['content']
for label in labels:
 a=read(r/'before-off'/f'{label}-response.json');b=read(r/'final-off'/f'{label}-response.json');assert a['tokens']==b['tokens'] and a['content']==b['content'],('native regression',label)
result['native_preserved']=True
manifest=read(r/'build-final-manifest.json');libhash=manifest['candidate_hashes']['libllama.so.0.4.0'];result['library_sha256']=libhash
for config in ['off','n1','n3']:
 m=read(r/('final-'+config)/'manifest.json');assert m['hashes']['libllama.so.0.4.0']==libhash
 assert m['loaded_libraries'][str(r/'final-bin/libllama.so.0.4.0')]==libhash
 result.setdefault('gpu_memory_mib',{})[config]={k:sum(int(line.split(',')[-1]) for line in m[k]) for k in ['gpu_memory_ready','gpu_memory_end']}
for name,sha in manifest['source_changes'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha,name
reg=read(r/'final-regression.json');assert reg['library_sha256']==libhash;assert len(reg['cases'])==7 and all(c['exit_code']==0 for c in reg['cases']);result['regressions']=7
replay_manifest=read(r/'final-replay-manifest.json');assert replay_manifest['lib']==libhash and all(c['exit_code']==0 for c in replay_manifest['cases'])
assert replay_manifest['harness_source_sha256']==hashlib.sha256((r/'replay.cpp').read_bytes()).hexdigest()
result['replay']={}
for name in ['final-real-replay','final-cpu-f16','final-cpu-q8','final-gpu-f16-replay','final-gpu-q8-replay']:
 rows=[json.loads(s) for s in (r/(name+'.jsonl')).read_text().splitlines()];assert len(rows)==5 and all(x['differing']==0 and x['max_abs']==0 for x in rows);result['replay'][name]=rows
(r/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for x in result['final']:print(x['case'],*[f"{c} {x[c]['median_tps']:.2f}" for c in ['off','n1','n3']])
print('All checks passed')
