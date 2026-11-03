from pathlib import Path
import json,re
P=Path(__file__).parent;r=json.load(open(P/'fresh-on-native/results.json'));m=json.load(open(P/'fresh-on-manifest.json'));old=json.load(open(P/'long-native/results.json'));pool=json.load(open(P/'fresh-on-trace.json.pool.json'))
assert r['complete']and m['exit_code']==0 and len(r['cases'])==12 and len(r['faults'])==2
assert (m['complete']or(m.get('error')=='AssertionError()'and "P/'long-native/results.json'" in(P/'run-fresh-on.measured.py').read_text()))
assert m['environment']['LLAMA_DSV41_CANDIDATE_POOL_LOCAL']=='1'
for row in r['cases']:
 ref=next(x['result']for x in old['cases']if x['corpus']==row['corpus']and x['mode']==0 and x['key']==row['key']);assert row['result']==ref,(row['corpus'],row['key'])
text=(P/'fresh-on.log').read_text(errors='replace');warnings=[l for l in text.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)];assert not warnings,warnings
inside=False;faults=0;bad=[]
for l in text.splitlines():
 if l=='FAULT_BEGIN':assert not inside;inside=True;faults+=1
 elif l=='FAULT_END':assert inside;inside=False
 elif re.search(r'\bE ',l)and not inside:bad.append(l)
assert not inside and faults==2 and not bad,bad
assert pool['local'][1]>0 and pool['remote'][1]==0,pool
minimum=min(x['min_free']/1048576 for ph in ['load','inference']for x in m['memory'][ph].values()if isinstance(x,dict)and'min_free'in x);assert minimum>=512 and not m.get('floor_breach')and not m.get('forced_kill')
s={'complete':True,'scope':'Fresh constantON from constructor, K5/raw sampling OFF. Two98K corpora,12raw result groups exact to prior mutable fixture OFF reference,including target/draft logits,state,taps,rollback,NLL128;two seed failures/retry. No shutdown compute-buffer warning.','groups':12,'raw_manifest_complete':m['complete'],'raw_postprocess_error':m.get('error'),'postprocess_correction':'Measured runner read the old36-group result path while expecting12. Native command/output path was fresh-on-native; corrected validator reads that file. Raw manifest/logs not rewritten.','seed_failures':r['faults'],'pool_placement':pool,'min_free_mib':minimum,'exit_code':m['exit_code'],'warnings':warnings,'compute_reserve_lines':[l for l in text.splitlines()if'CUDA2 compute buffer size ='in l]};(P/'fresh-on-summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
