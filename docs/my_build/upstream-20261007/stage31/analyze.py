from pathlib import Path
import hashlib,json,statistics,struct
R=Path(__file__).resolve().parent;S30=R.parent/'stage30'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text())
def equal(a,b):return {k:a.get(k)==b.get(k) for k in ['tokens','content','completion_probabilities']}
results={'server_cases':{},'fixture_cases':{},'limitations':['Comparisons of returned speculative probability fields cover only nonempty API rows, not all internal logits.','Parallel waves check successful completion; cross-schedule numerical equality is not required.','File restore checks use the generated tiny GLM5NEXT fixture, not the full model.']}
compact={}
for label,base,kind in [('late-native','preclear-native',0),('final-native','preclear-native',0),('final-late-mtp','preclear-mtp',1),('final-late-dflash','preclear-dflash',1)]:
 d=R/'server-runs'/label/f'glm5next-spec-{kind}';prev=S30/'server-runs'/base/f'glm5next-spec-{kind}'
 if not (d/'result.json').exists():continue
 row={'status':load(d/'result.json').get('status'),'prompts':[],'requests':len(load(d/'result.json').get('requests',[])),'RAM_hits':0,'parallel':[]}
 for i in range(3):
  p=d/f'fresh1-{i}-response.json'
  if not p.exists():continue
  a=load(p);comp={'index':i,'baseline':equal(a,load(prev/p.name)),'repeats':{}}
  for phase in ['reuse1','reuse2','fresh2']:
   p=d/f'{phase}-{i}-response.json'
   if not p.exists():continue
   b=load(p);comp['repeats'][phase]=equal(a,b);comp['repeats'][phase]['cache_n']=b['timings']['cache_n'];row['RAM_hits']+=int(phase.startswith('reuse') and b['timings']['cache_n']>0)
  row['prompts'].append(comp)
 for p in sorted(d.glob('parallel*-response.json')):
  x=load(p);row['parallel'].append({'name':p.name,'tokens':len(x.get('tokens',[])),'error':x.get('error')})
 for p in sorted(d.glob('*-response.json')):
  x=load(p);compact[str(p.relative_to(R))]={'sha256':sha(p),'tokens':x.get('tokens'),'content':x.get('content'),'timings':x.get('timings'),'probabilities_sha256':hashlib.sha256(json.dumps(x.get('completion_probabilities'),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),'nonempty_probability_rows':sum(bool(z.get('top_logprobs',z.get('probs',[]))) for z in x.get('completion_probabilities',[]))}
 fresh=[load(p)['timings']['predicted_per_second'] for p in d.glob('fresh*-response.json')];old=[load(p)['timings']['predicted_per_second'] for p in prev.glob('fresh*-response.json')]
 row['fresh_decode_median_tps']=statistics.median(fresh) if fresh else None;row['stage30_fresh_decode_median_tps']=statistics.median(old)
 results['server_cases'][label]=row
for kind in ['cpu','gpu']:
 d=R/kind/'ordered';pairs=[]
 if not d.exists():continue
 for seq in range(3):
  a=(d/f'phase0-seq{seq}.f32').read_bytes()
  for phase in range(1,4):
   b=(d/f'phase{phase}-seq{seq}.f32').read_bytes();pairs.append({'seq':seq,'phase':phase,'equal':a==b})
 results['fixture_cases'][kind]={'comparisons':pairs,'library_sha256':load(d/'manifest.json')['library_sha256']}
fa=load(R/'fa-results.json');results['frozen_FA']={'total_runs':len(fa),'ordered_runs':sum(r['ordered'] for r in fa),'ordered_all_exact':all(r['equal_ordered_reference'] for r in fa if r['ordered'])}
for f in [R/'layout-restore-gpu0.json',R/'layout-restore-gpu1.json']:
 if f.exists():results[f.stem]=load(f)
results['regression']=load(R/'regression-results.json')
results['final_source_hashes']=load(R/'build-manifest.json')['source_changes']
results['completed']=len(results['server_cases'])==4 and all(r['status']=='passed' and r['requests']==18 for r in results['server_cases'].values()) and (R/'layout-restore-gpu1.json').exists()
(R/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');(R/'responses-compact.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n')
for label,r in results['server_cases'].items():
 print(label,r['status'],r['requests'],'RAM',r['RAM_hits'],'equal',all(all(v[k] for k in ['tokens','content','completion_probabilities']) for p in r['prompts'] for v in p['repeats'].values()),flush=True)
