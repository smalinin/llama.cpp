from pathlib import Path
import json
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';A=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts/precision-0/results.json';ref={(x['corpus'],x['key']):x['result']for x in json.loads(A.read_text())['cases']}
def dif(a,b,path):
 if isinstance(a,dict)and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):yield from dif(a.get(k),b.get(k),path+'.'+k)
 elif isinstance(a,list)and isinstance(b,list):
  if len(a)!=len(b):yield {'path':path+'.length','before':len(a),'after':len(b)}
  for i,(x,y)in enumerate(zip(a,b)):yield from dif(x,y,path+f'[{i}]')
 elif a!=b:yield {'path':path,'reference':a,'actual':b}
case=json.loads((P/'precision-repeat-1/results.json').read_text());rows=[]
for row in case['cases']:rows.extend(dif(ref[row['corpus'],row['key']],row['result'],row['corpus']+'.'+row['key']))
(P/'repeat-differences.json').write_text(json.dumps(rows,indent=2)+'\n')
a=json.loads((P/'precision-results.json').read_text());b=json.loads((P/'precision-repeat-results.json').read_text());ea=a['arms'][0]['environment'];eb=b['arms'][0]['environment'];j={'source_hashes_equal':a['config']['source_files']==b['config']['source_files'],'module_hashes_equal':a['config']['source_library_hashes']==b['config']['source_library_hashes'],'expanded_model_args_equal':a['arms'][0]['command'][3:]==b['arms'][0]['command'][3:],'environment_differences':{k:[ea.get(k),eb.get(k)]for k in sorted(set(ea)|set(eb))if ea.get(k)!=eb.get(k)},'differences':len(rows),'earliest_recorded_difference':'first8K prefill taps/state, before target six-query branch is eligible; first prefix uses512-token ubatches and final4queries only. This does not prove whether the initial rollback failure has the same cause. Fresh OFF repeat pending.'};(P/'repeat-identity-check.json').write_text(json.dumps(j,indent=2)+'\n')
print('recorded',len(rows),'differences')
