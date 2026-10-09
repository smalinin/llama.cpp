from pathlib import Path
import json,random,struct,hashlib
R=Path(__file__).resolve().parent;S=R.parent/'stage30';out=R/'fa-inputs';out.mkdir(exist_ok=True)
source=Path('/home/sergei/Github/llama.cpp/src/llama-graph.cpp').read_text();a=source.index('static void build_attn_ordered(');b=source.index('\nggml_tensor * llm_graph_context::build_attn(',a);helper=source[a:b]
helper=helper.replace('const llm_graph_input_attn_k * inp','ggml_tensor * order, ggml_tensor * valid').replace('inp->self_kv_order','order').replace('inp->self_kv_valid','valid')
(R/'ordered-helper.inc').write_text(helper)
ref=(S/'fa-native-tensors/0-src1.bin').read_bytes();qm=(S/'fa-native-tensors/0-src3.bin').read_bytes();known={ref[j*1024:(j+1)*1024]:j for j in range(435)}
manifest=[]
def save(label,q,k,m,mapping):
 p=out/label;n=len(k)//1024;order=[0]*512
 for physical,logical in mapping.items():order[logical]=physical
 for name,data in [('src0',q),('src1',k),('src3',m),('params',(S/'fa-native-tensors/0-params.bin').read_bytes()),('order',struct.pack('<512i',*order)),('valid',struct.pack('<512f',*([0.0]*435+[float('-inf')]*77)))]:Path(str(p)+'-'+name+'.bin').write_bytes(data)
 manifest.append({'label':label,'physical_nkv':n,'live_rows':435,'logical_nkv':512,'physical_to_logical':mapping})
for i in range(4):
 k=(S/f'fa-native-tensors/{i}-src1.bin').read_bytes();m=(S/f'fa-native-tensors/{i}-src3.bin').read_bytes();n=len(k)//1024;mapping={}
 for j in range(n):
  if struct.unpack_from('<e',m,2*(3*n+j))[0]==0:mapping[j]=known[k[j*1024:(j+1)*1024]]
 assert len(mapping)==435
 save('capture'+str(i),(S/f'fa-native-tensors/{i}-src0.bin').read_bytes(),k,m,mapping)
q=(S/'fa-native-tensors/0-src0.bin').read_bytes()
for label,n,physical in [('shift',2048,list(range(911,911+435))),('wrap',4096,[(3900+i)%4096 for i in range(435)]),('holes',8192,[i*17 for i in range(435)]),('shuffle',4096,random.Random(42).sample(range(4096),435))]:
 k=bytearray(n*1024);m=bytearray(struct.pack('<e',float('-inf'))*(n*4));mapping={}
 for logical,p in enumerate(physical):
  mapping[p]=logical;k[p*1024:(p+1)*1024]=ref[logical*1024:(logical+1)*1024]
  for t in range(4):m[2*(t*n+p):2*(t*n+p+1)]=qm[2*(t*512+logical):2*(t*512+logical+1)]
 save(label,q,k,m,mapping)
(R/'fa-input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
