from pathlib import Path
import argparse,json,re,struct,math
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('label');a=p.parse_args();r=R/'server-runs'/a.label;log=next(r.rglob('server.log')).read_text();d=R/(a.label+'-tensors');items=[]
pattern=r'STAGE30_TENSOR id=(\d+) name=(\S+) op=(\S+) type=(\d+) ne=(\S+) nb=(\S+) bytes=(\d+)'
meta={}
for m in re.finditer(pattern,log):
 i,name,op,typ,ne,nb,nbytes=m.groups();meta[int(i),name]=dict(op=op,type=int(typ),ne=list(map(int,ne.split(','))),nb=list(map(int,nb.split(','))),bytes=int(nbytes))
for (i,name),v in meta.items():
 if i==0:continue
 ref=meta.get((0,name));item=dict(capture=i,name=name,**v)
 if not ref:item['missing_reference']=True;items.append(item);continue
 aa=(d/f'0-{name}.bin').read_bytes();bb=(d/f'{i}-{name}.bin').read_bytes();item['equal_bytes']=aa==bb
 if len(aa)==len(bb) and v['type'] in (0,1) and v['ne']==ref['ne'] and v['nb']==ref['nb']:
  fmt='f' if v['type']==0 else 'e';xs=struct.iter_unpack(fmt,aa);ys=struct.iter_unpack(fmt,bb);diff=[]
  for (x,),(y,) in zip(xs,ys):
   if math.isfinite(x) and math.isfinite(y):diff.append(abs(x-y))
  item['max_abs']=max(diff,default=0);item['rms']=math.sqrt(sum(x*x for x in diff)/len(diff)) if diff else None
 items.append(item)
(r/'tensor-comparison.json').write_text(json.dumps(items,indent=2)+'\n')
for i in sorted({k[0] for k in meta if k[0]}):
 bad=[x for x in items if x['capture']==i and not x.get('equal_bytes')];print('CAPTURE',i,'first',json.dumps(bad[:5]));print('CHANGED',[(x['name'],x.get('max_abs')) for x in bad])
