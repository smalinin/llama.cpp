from pathlib import Path
import json,re
P=Path(__file__).resolve().parent
r=json.loads((P/'lifetime-native/results.json').read_text());m=json.loads((P/'lifetime-manifest.json').read_text())
assert r['complete'] and m['complete'] and m['exit_code']==0 and not r.get('error') and len(r['cases'])==12
by={}
for c in r['cases']:by.setdefault((c['corpus'],c['n']),{})[c['mode']]=c['result']
assert len(by)==4
for k,v in by.items():assert set(v)=={0,1,2} and v[0]==v[1]==v[2],k
log=(P/'lifetime.log').read_text(errors='replace');inside=False;negative=0;full=0;unexpected=[]
for line in log.splitlines():
 if line=='NEGATIVE_BEGIN':assert not inside;inside=True;negative+=1
 elif line=='NEGATIVE_END':assert inside;inside=False
 elif re.search(r'\bE ',line):
  if 'full layer inputs unavailable; use the declared suffix' in line:full+=1
  elif not (inside and ('n_tokens == 0' in line or 'llama_decode: failed to decode, ret = -1' in line)):unexpected.append(line)
assert not inside and negative==12 and full==4 and not unexpected,unexpected
assert len(re.findall(r'prefill_skip: source=',log))==4
assert len(re.findall(r'dspark_prefill_tail: tokens=',log))==4
assert not re.search('does not match expectation|out of memory|failed to allocate|CUDA error|FAIL ',log,re.I)
floor=min(v['min_free']for v in m['memory']['inference'].values()if isinstance(v,dict)and 'min_free'in v)/1048576
assert floor>=512 and not m['memory'].get('error')
s={'complete':True,'cases':12,'active_prefixes':4,'target_draft_taps_K3_K5_verification_exact':True,'stale_reads_rejected_after':['seq restore','failed empty decode','registration mode change'],'fresh_reads_work_after_verify':True,'expected_empty_decode_errors':12,'expected_partial_full_getter_rejections':4,'min_free_mib':floor,'exit_code':0}
(P/'lifetime-summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
