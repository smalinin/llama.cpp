from pathlib import Path
import json,hashlib,array,math,struct,sys
assert sys.byteorder=='little'
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-forensic-20261105/artifacts';B=P/'bundle/blobs'
def scalar(a,b,typ):
 fmt={'f32':'f','f16':'e','bf16':None}[typ];size=4 if typ=='f32'else 2
 def values(x):
  if typ=='bf16':return[struct.unpack('<f',struct.pack('<I',v[0]<<16))[0]for v in struct.iter_unpack('<H',x)]
  return[v[0]for v in struct.iter_unpack('<'+fmt,x)]
 x=values(a);y=values(b);n=min(len(x),len(y));count=0;first=[];max_abs=0.;finite=0
 for i in range(n):
  if a[i*size:(i+1)*size]==b[i*size:(i+1)*size]:continue
  count+=1
  if len(first)<16:first.append(i)
  if math.isfinite(x[i])and math.isfinite(y[i]):max_abs=max(max_abs,abs(x[i]-y[i]));finite+=1
 return{'dtype':typ,'reference_elements':len(x),'actual_elements':len(y),'changed_words':count,'first_changed_elements':first,'max_abs_finite':max_abs,'changed_finite_pairs':finite}
def q8(a,b):
 assert len(a)%34==len(b)%34==0;count=0;first=[];max_abs=0.;changed_vals=0
 for off in range(0,min(len(a),len(b)),34):
  x=a[off:off+34];y=b[off:off+34]
  if x==y:continue
  count+=1
  if len(first)<16:first.append(off//34)
  dx,dy=struct.unpack('<e',x[:2])[0],struct.unpack('<e',y[:2])[0]
  for u,v in zip(x[2:],y[2:]):
   ux=(u if u<128 else u-256)*dx;vy=(v if v<128 else v-256)*dy
   if ux!=vy:changed_vals+=1
   if math.isfinite(ux)and math.isfinite(vy):max_abs=max(max_abs,abs(ux-vy))
 return{'dtype':'q8_0','changed_blocks':count,'first_changed_blocks':first,'changed_dequantized_values':changed_vals,'max_abs_finite':max_abs}
def bytes_diff(a,b):
 count=0;first=[]
 for off in range(0,min(len(a),len(b)),4096):
  x=a[off:off+4096];y=b[off:off+4096]
  if x==y:continue
  for i,(u,v)in enumerate(zip(x,y)):
   if u!=v:
    count+=1
    if len(first)<32:first.append({'offset':off+i,'reference':u,'actual':v})
 return{'reference_bytes':len(a),'actual_bytes':len(b),'changed_common_bytes':count,'first_changed_bytes':first}
def components(ref,row,key):
 aa=ref['descriptors'][key]['parts'];bb=row['descriptors'][key]['parts'];out=[]
 with(B/(ref['value'][key]+'.bin')).open('rb')as af,(B/(row['value'][key]+'.bin')).open('rb')as bf:
  for i,(u,v)in enumerate(zip(aa,bb)):
   if u['sha256']==v['sha256']:continue
   af.seek(u['offset']);bf.seek(v['offset']);a=af.read(u['bytes']);b=bf.read(v['bytes']);d={'index':i,'reference':u,'actual':v,'bytes':bytes_diff(a,b)}
   if u['kind']==v['kind']=='tensor'and u['type']==v['type']and len(a)==len(b):
    typ=u['type'];d['numeric']=scalar(a,b,typ)if typ in ['f32','f16','bf16']else q8(a,b)if typ=='q8_0'else{'dtype':typ,'not_decoded':True}
   out.append(d)
 return{'reference_parts':len(aa),'actual_parts':len(bb),'changed':out}
def main():
 run=json.loads((P/'results.json').read_text());paths=[P/f"arm-{a['index']}/results.json"for a in run['arms']];reference=None;out={'complete':run['complete'],'scope':'Raw first8K state/tap differences, fixedOFF; no numerical normalization or native speed claim.','cases':[]}
 for index,path in enumerate(paths):
  if not path.exists():continue
  try:data=json.loads(path.read_text())
  except json.JSONDecodeError:continue
  for row in data.get('cases',[]):
   if 'descriptors'not in row:continue
   if reference is None:reference=row
   j={'arm':index,'repeat':row['repeat'],'exact':row['value']==reference['value'],'tap_raw_matches_original':row['tap_raw_matches_original'],'metadata_consistent':row['metadata_consistent'],'states':{},'taps':[]}
   for k in ['target_seq','target_global','draft_seq','draft_global']:
    if row['value'][k]!=reference['value'][k]:j['states'][k]=components(reference,row,k)
   for i,(a,b)in enumerate(zip(reference['raw_tap_hashes'],row['raw_tap_hashes'])):
    if a!=b:j['taps'].append({'layer':37+i,'numeric':scalar((B/(a+'.bin')).read_bytes(),(B/(b+'.bin')).read_bytes(),'f32')})
   out['cases'].append(j)
 out['observed_reference_exact']=all(x['exact']for x in out['cases']);(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print([(x['arm'],x['repeat'],x['exact'],list(x['states']),[t['layer']for t in x['taps']])for x in out['cases']])
if __name__=='__main__':main()
