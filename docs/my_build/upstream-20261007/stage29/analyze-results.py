#!/usr/bin/env python3
"""Compare fixed request schedules and full CPU logits, without tolerating mismatches away."""
from pathlib import Path
import array,hashlib,json,math
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def floats(a,b):
 x=array.array('f');x.frombytes(a.read_bytes());y=array.array('f');y.frombytes(b.read_bytes())
 assert len(x)==len(y) and len(x)%128==0 and all(math.isfinite(v) for v in [*x,*y])
 delta=[abs(v-w) for v,w in zip(x,y)];rows=len(x)//128
 return {'bit_exact':a.read_bytes()==b.read_bytes(),'rows':rows,'max_abs':max(delta),'rms':math.sqrt(sum(v*v for v in delta)/len(delta)),'first_float_difference':next((i for i,v in enumerate(delta) if v),None),'greedy_equal':all(max(range(128),key=lambda i:x[r*128+i])==max(range(128),key=lambda i:y[r*128+i]) for r in range(rows))}
def compare(a,b):
 fields={k:a.get(k)==b.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated']}
 ta=a['tokens'];tb=b['tokens'];first=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y),min(len(ta),len(tb)) if len(ta)!=len(tb) else None)
 pa=a.get('completion_probabilities');pb=b.get('completion_probabilities');pfirst=None
 if pa is not None and pb is not None:pfirst=next((i for i,(x,y) in enumerate(zip(pa,pb)) if x!=y),min(len(pa),len(pb)) if len(pa)!=len(pb) else None)
 return {'exact_answer':all(fields.values()),'fields':fields,'first_token_difference':first,'probabilities_exact':pa==pb if pa is not None and pb is not None else None,'nonempty_probability_rows':[[i for i,row in enumerate(p or []) if row.get('top_logprobs')] for p in [pa,pb]],'first_probability_difference':pfirst,'cache_n':[a['timings'].get('cache_n',0),b['timings'].get('cache_n',0)],'prompt_n':[a['timings'].get('prompt_n'),b['timings'].get('prompt_n')],'predicted_n':[len(ta),len(tb)]}
out={'cpu':{},'cpu_pool_on_off':{},'server':{},'server_pool_on_off':{}}
manifest=read(R/'cpu-manifest.json');assert all(x['exit_code']==0 for x in manifest['cases'].values())
for label in manifest['cases']:
 d=R/'cpu'/label;out['cpu'][label]={}
 for a,b in [('resident','restore1'),('restore1','restore2'),('resident','new-context'),('tail-resident','tail-rollback'),('tail-resident','tail-restore')]:out['cpu'][label][a+'_vs_'+b]=floats(d/(a+'.f32'),d/(b+'.f32'))
for kind in ['unified','separated']:
 out['cpu_pool_on_off'][kind]={f.stem:floats(f,R/'cpu'/(kind+'-off')/f.name) for f in sorted((R/'cpu'/(kind+'-on')).glob('*.f32'))}
for control in sorted((R/'server-runs').iterdir()):
 ds=list(control.glob('glm5next-spec-*'));assert len(ds)==1;d=ds[0]
 if not (d/'result.json').exists():continue
 result=read(d/'result.json');entry={'status':result['status'],'requests':len(result['requests']),'server_exit_code':result.get('server_exit_code'),'comparisons':{}}
 out['server'][control.name]=entry
 if result['status']!='passed':continue
 n_prompts=3 if '-ram' in control.name else 4
 assert len(result['requests'])==4*n_prompts and result['server_exit_code']==0
 entry['cache_hits']=[{'phase':q['phase'],'prompt':q['prompt'],'cache_n':q['timings'].get('cache_n',0),'prompt_n':q['timings'].get('prompt_n')} for q in result['requests'] if q['timings'].get('cache_n',0)>0]
 entry['slots_per_prompt']={str(i):[q['slot'] for q in result['requests'] if q['prompt']==i] for i in range(n_prompts)}
 probability_rows=[row for f in d.glob('*-response.json') for row in read(f).get('completion_probabilities',[])]
 entry['probability_coverage']={'objects':len(probability_rows),'nonempty_top5':sum(bool(row.get('top_logprobs')) for row in probability_rows)}
 for i in range(n_prompts):
  entry['comparisons'][str(i)]={}
  for a,b in [('fresh1','fresh2'),('fresh1','reuse1'),('reuse1','reuse2')]:
   entry['comparisons'][str(i)][a+'_vs_'+b]=compare(read(d/f'{a}-{i}-response.json'),read(d/f'{b}-{i}-response.json'))
for kind in ['dflash','mtp','native']:
 if all(out['server'].get(kind+'-'+x,{}).get('status')=='passed' for x in ['on','off']):
  a=next((R/'server-runs'/(kind+'-on')).glob('glm5next-spec-*'));b=next((R/'server-runs'/(kind+'-off')).glob('glm5next-spec-*'))
  out['server_pool_on_off'][kind]={f.stem:compare(read(f),read(b/f.name)) for f in sorted(a.glob('*-response.json'))}
out['ram_pool_on_off']={}
if all(out['server'].get(k,{}).get('status')=='passed' for k in ['dflash-ram','dflash-ram-off']):
 a=next((R/'server-runs/dflash-ram').glob('glm5next-spec-*'));b=next((R/'server-runs/dflash-ram-off').glob('glm5next-spec-*'))
 out['ram_pool_on_off']={f.stem:compare(read(f),read(b/f.name)) for f in sorted(a.glob('*-response.json'))}
(R/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
short={'cpu':out['cpu'],'server':{},'pool_on_off':{}}
for kind,v in out['server'].items():
 short['server'][kind]={'status':v['status'],'requests':v['requests'],'comparisons':{p:{k:{field:c[field] for field in ['exact_answer','first_token_difference','probabilities_exact','cache_n']} for k,c in pairs.items()} for p,pairs in v['comparisons'].items()}}
for kind,v in out['server_pool_on_off'].items():short['pool_on_off'][kind]={'exact_answers':sum(c['exact_answer'] for c in v.values()),'exact_probabilities':sum(c['probabilities_exact'] is True for c in v.values()),'total':len(v),'differences':{k:c for k,c in v.items() if not c['exact_answer']}}
print(json.dumps(short,indent=2))
