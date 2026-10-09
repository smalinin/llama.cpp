from pathlib import Path
import hashlib,json,struct
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');m=json.loads((R/'summary.json').read_text())
assert m['completed']
for label,r in m['server_cases'].items():
 assert r['status']=='passed' and r['requests']==18 and r['RAM_hits']==6,label
 assert len(r['parallel'])==6 and all(x['tokens']==64 and x['error'] is None for x in r['parallel'])
 assert len(r['prompts'])==3
 for p in r['prompts']:
  assert all(p['baseline'].values())
  assert len(p['repeats'])==3
  for x in p['repeats'].values():assert all(x[k] for k in ['tokens','content','completion_probabilities'])
for g in [0,1]:
 for x in m[f'layout-restore-gpu{g}']:
  assert x['exit_code']==0
  if x['build']=='after':assert len(x['rows'])==5 and all(y['logits_equal'] for y in x['rows'])
assert all(x['exit_code']==0 for x in m['regression'])
assert m['frozen_FA']['ordered_all_exact'] and m['frozen_FA']['ordered_runs']==32
for r in m['fixture_cases'].values():assert len(r['comparisons'])==9 and all(p['equal'] for p in r['comparisons'])
assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest()==h for n,h in m['final_source_hashes'].items())
comparisons=[]
for gpu in [0,1]:
 for q8 in [0,1]:
  arrays=[];raw=[];argmax=[]
  for kind in ['before','after']:
   b=(R/'layout-restore'/f'{kind}-gpu{gpu}-q{q8}'/'fresh.f32').read_bytes();a=struct.unpack('<%df'%(len(b)//4),b);arrays.append(a);raw.append(b);n=len(a)//4;argmax.append([max(range(n),key=lambda j:a[i*n+j]) for i in range(4)])
  comparisons.append({'gpu':gpu,'q8':q8,'fresh_before_after_equal':raw[0]==raw[1],'max_abs':max(abs(x-y) for x,y in zip(*arrays)),'argmax_equal':argmax[0]==argmax[1]})
assert all(x['argmax_equal'] for x in comparisons)
m.update(fresh_fixture_before_after=comparisons,validation_assertions_passed=True)
(R/'summary.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print('All result assertions and final source hashes verified')
