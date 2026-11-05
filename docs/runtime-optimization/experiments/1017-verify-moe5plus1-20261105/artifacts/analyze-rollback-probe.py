from pathlib import Path
import json
P=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts')
def changed(a,b):
 return [{"index":i,"before":x,"after":y}for i,(x,y)in enumerate(zip(a,b))if x!=y]
result={'arms':{},'cross_flag':[]}
for flag in ['0','1']:
 path=P/f'rollback-probe-{flag}/results.json'
 if not path.exists():continue
 data=json.loads(path.read_text());old=json.loads((P/'precision-inputs.json').read_text())
 arm={'complete':data['complete'],'cases':[x['key']for x in data['cases']],'rollback':[]}
 for row in data.get('state_debug',[]):
  dif=changed(row['before']['parts'],row['after']['parts'])
  arm['rollback'].append({k:row[k]for k in ['n','keep','phases','difference','first_vs_second_state_read_equal']}|{'parts_changed':dif,'metadata_descriptor_mismatches':row['before']['metadata_changed_while_describing']+row['after']['metadata_changed_while_describing']})
 result['arms'][flag]=arm
if len(result['arms'])==2:
 a=json.loads((P/'rollback-probe-0/results.json').read_text());b=json.loads((P/'rollback-probe-1/results.json').read_text())
 for x,y in zip(a.get('state_debug',[]),b.get('state_debug',[])):
  result['cross_flag'].append({'n':x['n'],'keep':x['keep'],'ids_equal':x['ids']==y['ids'],'phases_equal':x['phases']==y['phases'],'before_parts_changed':changed(x['before']['parts'],y['before']['parts']),'after_parts_changed':changed(x['after']['parts'],y['after']['parts'])})
print(json.dumps(result,indent=2))
