from pathlib import Path
import hashlib,json,re,struct
R=Path(__file__).resolve().parent;d=R/'fa-native-tensors';out=R/'fa-frozen';out.mkdir(exist_ok=True)
read=lambda i,n:(d/f'{i}-{n}.bin').read_bytes()
refk=read(0,'src1');refm=read(0,'src3');nr=len(refk)//1024
refrows={refk[j*1024:(j+1)*1024]:j for j in range(nr) if struct.unpack_from('<e',refm,2*(3*nr+j))[0]==0};assert len(refrows)==435
summary=[]
for i in range(4):
 k=read(i,'src1');m=read(i,'src3');n=len(k)//1024;mapping={};missing=[]
 for j in range(n):
  if struct.unpack_from('<e',m,2*(3*n+j))[0]==0:
   old=refrows.get(k[j*1024:(j+1)*1024])
   if old is None:missing.append(j)
   else:mapping[j]=old
 row={'capture':i,'nkv':n,'q_equal':read(i,'src0')==read(0,'src0'),'visible_key_rows':len(mapping)+len(missing),'unmatched_visible_keys':missing,'physical_mapping':mapping}
 cm=bytearray(struct.pack('<e',float('-inf'))*(nr*4));ck=bytearray(refk)
 if not missing:
  for j,old in mapping.items():
   ck[old*1024:(old+1)*1024]=k[j*1024:(j+1)*1024]
   for q in range(4):cm[2*(q*nr+old):2*(q*nr+old+1)]=m[2*(q*n+j):2*(q*n+j+1)]
  row.update(canonical_keys_equal=ck==refk,canonical_mask_equal=cm==refm)
  for name,data in [('src0',read(i,'src0')),('src1',ck),('src2',ck),('src3',cm),('params',read(0,'params')),('fa',read(0,'fa'))]:(out/f'canonical{i}-{name}.bin').write_bytes(data)
 summary.append(row)
(R/'fa-input-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
for x in summary:print({k:v for k,v in x.items() if k!='physical_mapping'})
