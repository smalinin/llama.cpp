from pathlib import Path
import json,hashlib,collections
import numpy as np
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-shared-prefix-20261105/artifacts';B=P/'bundle/blobs'
def vector(a,b):
 x=np.fromfile(B/(a+'.bin'),dtype='<f4');y=np.fromfile(B/(b+'.bin'),dtype='<f4');j={'count_reference':int(x.size),'count_actual':int(y.size)}
 if x.shape==y.shape:
  finite=np.isfinite(x)&np.isfinite(y);j.update(changed_words=int(np.count_nonzero(x.view('<u4')!=y.view('<u4'))),max_abs_finite=float(np.max(np.abs(x[finite].astype('f8')-y[finite].astype('f8'))))if np.any(finite)else None,nonfinite_reference=int(np.count_nonzero(~np.isfinite(x))),nonfinite_actual=int(np.count_nonzero(~np.isfinite(y))))
 return j

def components(a,b):
 x=a['state_parts']['parts'];y=b['state_parts']['parts'];out=[]
 for i,(u,v)in enumerate(zip(x,y)):
  if u['sha256']==v['sha256']:continue
  q={'index':i,'reference':u,'actual':v}
  if u['kind']==v['kind']=='tensor'and u['type']==v['type']=='f32'and u['bytes']==v['bytes']:
   aa=np.memmap(B/(a['value']['target']+'.bin'),mode='r',dtype='<f4',offset=u['offset'],shape=(u['bytes']//4,));bb=np.memmap(B/(b['value']['target']+'.bin'),mode='r',dtype='<f4',offset=v['offset'],shape=(v['bytes']//4,));changed=aa.view('<u4')!=bb.view('<u4');f=np.isfinite(aa)&np.isfinite(bb);q['numeric']={'changed_words':int(np.count_nonzero(changed)),'first_changed_indices':np.flatnonzero(changed)[:16].tolist(),'max_abs_finite':float(np.max(np.abs(aa[f].astype('f8')-bb[f].astype('f8'))))if np.any(f)else None}
  out.append(q)
 return out

def main():
 run=json.loads((P/'results.json').read_text());ref=json.loads((P/'arm-0-0/results.json').read_text());rmap={r['key']:r for r in ref['cases']};out={'complete':run['complete'],'arms':[],'scope':'Exact shared-prefix state/raw target comparisons; diagnostic counters, no native speed claim.'}
 for arm in run['arms']:
  path=P/f"arm-{arm['index']}-{arm['flag']}/results.json"
  if not path.exists():continue
  a=json.loads(path.read_text());groups=collections.defaultdict(list)
  for row in a.get('cases',[]):
   v=row['value'];groups[(row['key'].split('/')[0],v['n'],v.get('keep'))].append(v)
  j={'index':arm['index'],'flag':arm['flag'],'complete':a['complete'],'internal_exact':a['internal_exact'],'cases':len(a['cases']),'within_arm_repeats_exact':all(all(x==v[0]for x in v)for v in groups.values()),'changed_cases':[],'n6_moe_counts':dict(collections.Counter(row['counts']['moe_matmul']for row in a['cases']if row['value']['n']==6))}
  for row in a['cases']:
   rr=rmap[row['key']];changed=[k for k in set(rr['value'])|set(row['value'])if rr['value'].get(k)!=row['value'].get(k)]
   if not changed:continue
   d={'key':row['key'],'fields':changed,'components':components(rr,row)if'target'in changed else[],'vectors':{}}
   for k in ['logits','verify','tail']:
    if k in changed and(B/(row['value'][k]+'.bin')).exists():d['vectors'][k]=vector(rr['value'][k],row['value'][k])
   j['changed_cases'].append(d)
  if' memory'in arm:pass
  vals=[v['min_free']/2**20 for phase in ['load','inference']for v in arm.get('memory',{}).get(phase,{}).values()if isinstance(v,dict)and'min_free'in v]
  if vals:j['actual_floor_mib']=min(vals)
  out['arms'].append(j)
 (P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print([(a['index'],a['cases'],a['internal_exact'],a['within_arm_repeats_exact'],len(a['changed_cases']))for a in out['arms']])
if __name__=='__main__':main()
