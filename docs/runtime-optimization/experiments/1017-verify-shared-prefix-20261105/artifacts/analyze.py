from pathlib import Path
import json,hashlib,collections
import array,math,sys
assert sys.byteorder=="little"
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-shared-prefix-20261105/artifacts';B=P/'bundle/blobs'
def float_stats(a,b):
 x=array.array('f');x.frombytes(a);y=array.array('f');y.frombytes(b);xi=array.array('I');xi.frombytes(a);yi=array.array('I');yi.frombytes(b);j={'count_reference':len(x),'count_actual':len(y)}
 if len(x)==len(y):
  changes=[i for i,(u,v)in enumerate(zip(xi,yi))if u!=v];delta=[abs(float(u)-float(v))for u,v in zip(x,y)if math.isfinite(u)and math.isfinite(v)];j.update(changed_words=len(changes),first_changed_indices=changes[:16],max_abs_finite=max(delta)if delta else None,nonfinite_reference=sum(not math.isfinite(u)for u in x),nonfinite_actual=sum(not math.isfinite(u)for u in y))
 return j
def vector(a,b):return float_stats((B/(a+'.bin')).read_bytes(),(B/(b+'.bin')).read_bytes())

def components(a,b):
 x=a['state_parts']['parts'];y=b['state_parts']['parts'];out=[]
 for i,(u,v)in enumerate(zip(x,y)):
  if u['sha256']==v['sha256']:continue
  q={'index':i,'reference':u,'actual':v}
  if u['kind']==v['kind']=='tensor'and u['type']==v['type']=='f32'and u['bytes']==v['bytes']:
   with (B/(a['value']['target']+'.bin')).open('rb')as af,(B/(b['value']['target']+'.bin')).open('rb')as bf:
    af.seek(u['offset']);bf.seek(v['offset']);q['numeric']=float_stats(af.read(u['bytes']),bf.read(v['bytes']))
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
  vals=[v['min_free']/2**20 for phase in ['load','inference']for v in arm.get('memory',{}).get(phase,{}).values()if isinstance(v,dict)and'min_free'in v]
  if vals:j['actual_floor_mib']=min(vals)
  out['arms'].append(j)
 (P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print([(a['index'],a['cases'],a['internal_exact'],a['within_arm_repeats_exact'],len(a['changed_cases']))for a in out['arms']])
if __name__=='__main__':main()
