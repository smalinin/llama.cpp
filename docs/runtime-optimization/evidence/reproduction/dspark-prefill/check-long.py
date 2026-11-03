from pathlib import Path
import json,re,math
P=Path(__file__).resolve().parent;r=json.loads((P/'long-native/results.json').read_text());m=json.loads((P/'long-manifest.json').read_text());t=json.loads((P/'long-trace.json').read_text())
assert r['complete']and m['complete']and m['exit_code']==0 and not r.get('error')and len(r['cases'])==30 and len(r['faults'])==2
by={}
for c in r['cases']:by.setdefault((c['corpus'],c['key']),{})[c['mode']]=c['result']
assert len(by)==10
for k,v in by.items():assert set(v)=={0,1,2}and v[0]==v[1]==v[2],k
for (corpus,key),v in by.items():
 if key=='rollback':assert all(q['target_replay']and q['draft_replay']for rows in v.values()for q in rows)
for q in t['modes']:
 assert q['prefill_candidate_graphs']>0 and q['max_candidate_rows']>=98304,q
 assert (q['kv_only_graphs']>0)==(q['mode']==1),q
assert t['injected_failures']==2
log=(P/'long.log').read_text(errors='replace');inside=False;faults=0;unexpected=[]
for line in log.splitlines():
 if line=='FAULT_BEGIN':assert not inside;inside=True;faults+=1
 elif line=='FAULT_END':assert inside;inside=False
 elif re.search(r'\bE ',line)and not inside:unexpected.append(line)
assert not inside and faults==2 and not unexpected,unexpected
assert not re.search('does not match expectation|out of memory|failed to allocate|CUDA error',log,re.I)
floor=min(v['min_free']for v in m['memory']['inference'].values()if isinstance(v,dict)and 'min_free'in v)/1048576;assert floor>=512 and not m['memory'].get('error')
s={'complete':True,'cases':30,'faults':r['faults'],'rollback_cases':36,'candidate_trace':t,'min_free_mib':floor,'nll':[]}
for (corpus,key),v in by.items():
 if key=='nll128':s['nll'].append({'corpus':corpus,'off':v[0]['mean_nll'],'on':v[1]['mean_nll'],'explicit_off':v[2]['mean_nll'],'delta':v[1]['mean_nll']-v[0]['mean_nll'],'ppl':math.exp(v[0]['mean_nll']),'labels_per_mode':128})
(P/'long-summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
