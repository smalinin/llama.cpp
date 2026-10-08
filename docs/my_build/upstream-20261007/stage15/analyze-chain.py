from array import array
import ctypes,hashlib,json,math,struct
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'chain-capture-output'
MODE='decode-scalar-fa-upgate-hc-router'
metric=ctypes.CDLL(str(R.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read_tensor(directory,t,axis=None,column=0):
 raw=(directory/t['file']).read_bytes();ne=t['ne'].copy();offset=0
 if axis is not None:
  assert 0<=column<ne[axis]
  ne[axis]=1;offset=column*t['nb'][axis]
 a=array('f');kind=t['type']
 for i3 in range(ne[3]):
  for i2 in range(ne[2]):
   for i1 in range(ne[1]):
    start=offset+sum(i*b for i,b in zip([0,i1,i2,i3],t['nb']))
    if kind==0 and t['nb'][0]==4:
     a.frombytes(raw[start:start+ne[0]*4]);continue
    for i0 in range(ne[0]):
     at=start+i0*t['nb'][0]
     if kind==0:v=struct.unpack_from('<f',raw,at)[0]
     elif kind==1:v=struct.unpack_from('<e',raw,at)[0]
     elif kind==30:v=struct.unpack('<f',struct.pack('<I',struct.unpack_from('<H',raw,at)[0]<<16))[0]
     elif kind==26:v=struct.unpack_from('<i',raw,at)[0]
     else:raise ValueError(kind)
     a.append(v)
 return a

def compare(a,b):
 assert len(a)==len(b)
 o=(ctypes.c_double*5)();metric(a.buffer_info()[0],b.buffer_info()[0],len(a),0,o)
 assert o[4]==0
 return {'max_abs':o[2],'rms':math.sqrt(o[0]/len(a)),'bit_identical':a.tobytes()==b.tobytes()}

if __name__=='__main__':
 result={'controls':{},'captures':{}}
 for w in [1,2,4]:
  p=D/f'{MODE}-w{w}-logits.f32';old=R.parent/'stage14/extended-replay-output'/p.name
  result['controls'][str(w)]={'sha256':sha(p),'reference_sha256':sha(old),'bit_identical':sha(p)==sha(old)}
 assert all(c['bit_identical'] for c in result['controls'].values())
 for target in [0,86]:
  dirs={w:D/f'{MODE}-w{w}-input{target}' for w in [1,2,4]}
  tensors={w:[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()] for w,d in dirs.items()}
  alignment={w:json.loads((d/'alignment.json').read_text()) for w,d in dirs.items()}
  assert len(set(map(len,tensors.values())))==1
  records={str(w):[] for w in [2,4]};occ={}
  for ts in zip(*(tensors[w] for w in [1,2,4])):
   a,b,c=ts;assert len({(t['owner'],t['role']) for t in ts})==1
   key=(a['owner'],a['role']);index=occ.get(key,0);occ[key]=index+1
   axes=[i for i,(n,m) in enumerate(zip(a['ne'],b['ne'])) if n!=m]
   assert len(axes)<=1,(key,axes)
   axis=axes[0] if axes else None
   if axis is not None:assert a['ne'][axis]==1 and b['ne'][axis]==2 and c['ne'][axis]==4,(key,ts)
   arrays={w:read_tensor(dirs[w],t,axis,alignment[w]['column'] if axis is not None else 0) for w,t in zip([1,2,4],ts)}
   for w,t in [(2,b),(4,c)]:
    v=compare(arrays[1],arrays[w]);v.update(owner=a['owner'],role=a['role'],occurrence=index,op=a['op'],axis=axis,file1=a['file'],filew=t['file'],elements=len(arrays[1]))
    records[str(w)].append(v)
  result['captures'][str(target)]={'alignment':alignment,'counts':{w:len(ts) for w,ts in tensors.items()},'comparisons':records}
 (R/'chain-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'controls':result['controls'],'captures':{target:{'alignment':v['alignment'],'counts':v['counts'],'first_outputs':{w:[r for r in rs if r['role']=='output' and not r['bit_identical']][:15] for w,rs in v['comparisons'].items()},'first_operands':{w:[r for r in rs if r['role']!='output' and not r['bit_identical']][:10] for w,rs in v['comparisons'].items()}} for target,v in result['captures'].items()}},indent=2))
