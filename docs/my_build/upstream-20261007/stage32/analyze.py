from pathlib import Path
import json,hashlib,difflib,struct,math
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def cmp(a,b):
 pa,pb=a.get('completion_probabilities'),b.get('completion_probabilities')
 return {'tokens_equal':a['tokens']==b['tokens'],'text_equal':a['content']==b['content'],'first_token_difference':next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None),'probabilities_equal':pa==pb if pa else None,'tokens':len(a['tokens'])}
summary={'server':{},'fixtures':{},'fresh_before_after':{},'regression':read(R/'final-regression.json')['cases']}
for label in ['before','after','final']:
 d=R/label
 if not (d/'results.json').exists():continue
 def response(n):return read(d/(n+'-response.json'))
 control,restored=response('control'),response('restored');fresh,cached=response('long-fresh'),response('long-cache')
 assert control['timings']['cache_n']==restored['timings']['cache_n']==1686
 assert fresh['timings']['cache_n']==0 and cached['timings']['cache_n']==6613
 summary['server'][label]={'session':cmp(control,restored),'long':cmp(fresh,cached),'timings':{n:response(n)['timings'] for n in ['initial','control','restored','long-fresh','long-cache']},'binary':read(d/'manifest.json')['hashes']['libllama.so.0.4.0'],'graphs_disabled':read(d/'manifest.json')['env'].get('GGML_CUDA_DISABLE_GRAPHS')=='1'}
 if label!='before':
  assert summary['server'][label]['session']['probabilities_equal']
  assert summary['server'][label]['session']['tokens_equal'] and summary['server'][label]['long']['tokens_equal']
for name in ['initial','control','long-fresh','long-cache']:
 a=read(R/'before'/(name+'-response.json'));b=read(R/'after'/(name+'-response.json'));summary['fresh_before_after'][name]=cmp(a,b)
for gpu in [0,1]:
 d=R/f'final-gpu{gpu}';runs=read(d/'results.json');summary['fixtures'][f'gpu{gpu}']=runs
 for run in runs:
  assert run['exit_code']==0
  label=run['name']
  for row in run['rows']:
   n=row['prefix'];a=(d/label/f'{n}fresh.f32').read_bytes();b=(d/label/f'{n}restored.f32').read_bytes()
   assert len(a)==len(b)==8*128*4
   assert all(math.isfinite(x) for x in struct.unpack(f'{len(b)//4}f',b))
   assert row['exact']==(a==b)
   if label.startswith('after'):assert a==b
summary['small_fixture_cpu']=read(R/'cpu-results.json')
assert all(r['exit_code']==0 for r in summary['regression'])
base=read(R/'base-sources.json');patch=[];sources={}
for name,h in base.items():
 before=R/'base-src'/name.replace('/','_');assert sha(before)==h
 after=REPO/name
 if sha(after)!=h:
  patch.extend(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),'a/'+name,'b/'+name,n=0));sources[name]=sha(after)
(R/'source-final.patch').write_text(''.join(patch));summary['source_changes_since_stage31']=sources
build=read(R/'build-manifest.json');assert all(build['source_changes'][k]==v for k,v in sources.items());assert sha(R/'candidate-bin/libllama.so.0.4.0')==build['candidate_hashes']['libllama.so.0.4.0']
summary['final_library_sha256']=sha(R/'candidate-bin/libllama.so.0.4.0');summary['model_fixture_sha256']=sha(R/'models/deepseek41-moe.gguf');summary['fixture_exact_after']=sum(len(x['rows']) for runs in summary['fixtures'].values() for x in runs if x['name'].startswith('after'))
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('verified',summary['fixture_exact_after'],'fixture comparisons; server cases:',list(summary['server']),flush=True)
