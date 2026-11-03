from pathlib import Path
import json,re,math
P=Path(__file__).parent
r=json.load(open(P/'long-native/results.json'))
m=json.load(open(P/'long-manifest.json'))
t=json.load(open(P/'long-trace.json'))
assert r['complete']and m['complete']and m['exit_code']==0 and not r.get('error')and len(r['cases'])==36 and len(r['faults'])==2
by={}
for row in r['cases']:by.setdefault((row['corpus'],row['key']),{})[row['mode']]=row['result']
assert len(by)==12
for key,rows in by.items():assert set(rows)=={0,1,2}and rows[0]==rows[1]==rows[2],key
for (corpus,key),rows in by.items():
 if key=='rollback':assert all(x['target_replay']and x['draft_replay']for mode in rows.values()for x in mode)
assert t['aliases'][1]>0 and t['aliases'][2]>0
pool=json.load(open(P/'long-trace.json.pool.json'));assert pool['local'][1]>0 and pool['local'][2]==0 and pool['remote'][1]==0 and pool['remote'][2]>0,pool
assert t['injected_failures']==2
assert all(x['kv_only_graphs']>0 for x in t['modes']if x['mode']in[1,2])
inside=False;faults=0;unexpected=[]
for line in (P/'long.log').read_text(errors='replace').splitlines():
 if line=='FAULT_BEGIN':assert not inside;inside=True;faults+=1
 elif line=='FAULT_END':assert inside;inside=False
 elif re.search(r'\bE ',line)and not inside:unexpected.append(line)
assert not inside and faults==2 and not unexpected,unexpected
log=(P/'long.log').read_text(errors='replace')
assert not re.search('does not match expectation|out of memory|failed to allocate|CUDA error',log,re.I)
minimum=min(v['min_free']for phase in ['load','inference']for v in m['memory'][phase].values()if isinstance(v,dict)and'min_free'in v)/1048576
assert minimum>=512 and not m['memory'].get('error')and not m.get('forced_kill')and not m.get('floor_breach')
out={'complete':True,'result_groups':36,'rollback_workflows':36,'failures':r['faults'],'append_widths':[1,7,511,513],'trace':t,'pool_placement':pool,'sampled_min_free_mib':minimum,'nll':[]}
for (corpus,key),rows in by.items():
 if key=='nll128':
  diff=rows[1]['mean_nll']-rows[0]['mean_nll'];assert diff==0
  out['nll'].append({'corpus':corpus,'labels_per_mode':128,'off':rows[0]['mean_nll'],'on':rows[1]['mean_nll'],'delta':diff,'PPL_ratio':math.exp(diff)})
(P/'long-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print('LONG_VALIDATED',out['result_groups'],out['rollback_workflows'],minimum,out['nll'],flush=True)
