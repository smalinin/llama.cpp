import json
from pathlib import Path
cases=[];negative=0
for n in range(3073,8193):
 raw=768;u=512;window=128;last_tap=39;source=20
 begin=(n-raw)//u*u
 required=(last_tap-source-1)*(window-1)
 basecut=max(0,(n-3072)//u*u)
 cut=min(basecut,max(0,(begin-required)//u*u))
 if cut:
  assert n-begin>=raw and n-begin<raw+u
  assert begin%u==0 and cut%u==0
  assert cut+required<=begin
  assert cut+19*127<=n-128
  assert cut+18*127<=n-768
  if basecut+required>begin:negative+=1
  cases.append({'count':n,'cut':cut,'keep':n-cut,'first_valid':begin,'tail':n-begin})
j={'complete':True,'counts_checked':5120,'active_counts':len(cases),'old_cut_rejected':negative,'examples':[q for q in cases if q['count']in [4096,7676,7681,8191,8192]]}
Path(__file__).with_name('cone-results.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
