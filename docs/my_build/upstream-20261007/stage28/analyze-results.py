#!/usr/bin/env python3
from pathlib import Path
import argparse,array,hashlib,json,math,re
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
run=read(R/'pool-run-manifest.json');assert run['cases']['before-unified']['exit_code']==-6
assert 'incremental pool-key update capacity is too small' in (R/'before-unified.log').read_text()
summary={'cpu':{},'existing':read(R/'final-existing-tests.json'),'existing_v1':read(R/'existing-tests.json'),'server':{},'comparisons':{}}
assert all(x['exit_code']==0 for x in summary['existing'].values())
for name in ['after-unified','reference-unified','before-separated','after-separated']:
 assert run['cases'][name]['exit_code']==0
 rows=[json.loads(l) for l in (R/(name+'.jsonl')).read_text().splitlines() if l.startswith('{')]
 summary['cpu'][name]=rows[-1];assert rows[-1]['done'] and rows[-1]['calls']==21 and rows[-1]['rows']==46
for a,b in [('after-unified','reference-unified'),('before-separated','after-separated')]:
 assert (R/(a+'.f32')).read_bytes()==(R/(b+'.f32')).read_bytes()
 summary['comparisons'][a+'_vs_'+b]={'full_logits_bit_exact':True,'rows':46,'sha256':sha(R/(a+'.f32'))}
assert summary['cpu']['after-unified']['full']>0 and summary['cpu']['after-unified']['incremental']>0
for label in ['before-unified','after-unified','reference-unified','after-separated','mtp-unified','mtp-final-unified']:
 d=R/'server-runs'/label/'glm5next-spec-1'
 if not (d/'result.json').exists():
  assert args.partial,label;continue
 result=read(d/'result.json');summary['server'][label]=result
 if label in ['before-unified','mtp-unified']:
  assert result['status']=='failed' and result['server_exit_code']==-6 and len(result['requests'])==3
  assert ('incremental pool-key update capacity is too small' if label=='before-unified' else 'n_tps == 1') in (d/'server.log').read_text()
 else:assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==12
 for q in d.glob('*-request.json'):
  original=R.parent/'stage2/parallel/dflash-unified/glm5next-spec-1'/q.name
  if not original.exists():original=R.parent/'stage2/parallel/dflash-separated/glm5next-spec-1'/q.name
  assert read(q)==read(original),q
 if label not in ['before-unified','mtp-unified']:
  for q in d.glob('*-response.json'):
   response=read(q);assert response['tokens'] and len(response['tokens'])==response['tokens_predicted']<=64
   assert response.get('truncated') is False
if all(x in summary['server'] for x in ['after-unified','reference-unified']):
 controls=[]
 for q in sorted((R/'server-runs/after-unified/glm5next-spec-1').glob('*-response.json')):
  a=read(q);b=read(R/'server-runs/reference-unified/glm5next-spec-1'/q.name)
  fields={k:a.get(k)==b.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated']}
  first=next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None)
  controls.append({'case':q.stem,'equal':all(fields.values()),'fields':fields,'first_token_difference':first,
                   'candidate_cache_n':a['timings'].get('cache_n',0),'reference_cache_n':b['timings'].get('cache_n',0)})
 summary['comparisons']['server_pool_cache_off']={'pairs':len(controls),'exact':sum(c['equal'] for c in controls),'cases':controls}
if 'after-unified' in summary['server']:
 controls=[]
 for i in range(3):
  n=f'serial-{i}-response.json';a=read(R/'server-runs/before-unified/glm5next-spec-1'/n);b=read(R/'server-runs/after-unified/glm5next-spec-1'/n)
  controls.append({'case':n,'exact':all(a.get(k)==b.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated'])})
 summary['comparisons']['server_before_serial']=controls
for a,b,n in [('final-pool-unified','final-pool-reference',46),('final-pool-separated','before-separated',46),('mtp-final-unified','mtp-reference-unified',12),('mtp-final-separated','mtp-before-separated',12),('mtp-final-single','mtp-before-single',4)]:
 assert (R/(a+'.f32')).read_bytes()==(R/(b+'.f32')).read_bytes()
 summary['comparisons'][a+'_vs_'+b]={'full_logits_bit_exact':True,'rows':n,'sha256':sha(R/(a+'.f32'))}
for old,final in [('after-unified','final-pool-unified'),('reference-unified','final-pool-reference'),('after-separated','final-pool-separated')]:assert (R/(old+'.f32')).read_bytes()==(R/(final+'.f32')).read_bytes()
mtp=read(R/'mtp-run-manifest.json');assert mtp['cases']['mtp-before-unified']['exit_code']==-6
assert 'n_tps == 1' in (R/'mtp-before-unified.log').read_text()
summary['mtp_cpu']={}
for name,v in mtp['cases'].items():
 if name=='mtp-before-unified':continue
 assert v['exit_code']==0
 rows=[json.loads(l) for l in (R/(name+'.jsonl')).read_text().splitlines() if l.startswith('{')]
 assert rows[-1]['done'];summary['mtp_cpu'][name]=rows
 if name in ['mtp-final-separated','mtp-final-single']:assert rows[-2]['key']==rows[-3]['key']==rows[-4]['key']
 if name=='mtp-final-unified':assert rows[-2]['key']>rows[-3]['key']>rows[-4]['key']
summary['scope']='Capacity abort and pooling correctness. CPU fixed histories are bit-exact; concurrent server answer comparisons also depend on actual scheduling. No throughput claim or strict scheduler reallocation fix.'
name='partial-summary.json' if args.partial else 'summary.json';(R/name).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'cpu_comparisons':{k:v for k,v in summary['comparisons'].items() if 'server' not in k},'server_completed':{k:(v['status'],len(v['requests'])) for k,v in summary['server'].items()},'server_comparisons':{k:v for k,v in summary['comparisons'].items() if 'server' in k}},indent=2))
