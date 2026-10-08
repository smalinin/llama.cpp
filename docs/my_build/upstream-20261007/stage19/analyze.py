from pathlib import Path
from array import array
import ctypes,hashlib,json,math

R=Path(__file__).resolve().parent
MODE='decode-scalar-fa-upgate-hc-router-down-compressor'
metric=ctypes.CDLL(str(R.parent/'stage5/metrics.so')).stage5_metrics
metric.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
VOCAB=129280
def sha(p,limit=None):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while limit is None or limit>0:
   b=f.read(min(4*1024*1024,limit) if limit is not None else 4*1024*1024)
   if not b:break
   h.update(b)
   if limit is not None:limit-=len(b)
 return h.hexdigest()

def compare(a,b):
 assert a.stat().st_size==b.stat().st_size
 rows=a.stat().st_size//(VOCAB*4)
 assert rows*VOCAB*4==a.stat().st_size
 total=maximum=0.0;first=None;different=[]
 with a.open('rb') as fa,b.open('rb') as fb:
  for i in range(rows):
   aa=fa.read(VOCAB*4);bb=fb.read(VOCAB*4)
   if aa!=bb:
    if first is None:first=i
    different.append(i)
   x=array('f');x.frombytes(aa);y=array('f');y.frombytes(bb)
   o=(ctypes.c_double*5)();metric(x.buffer_info()[0],y.buffer_info()[0],VOCAB,0,o)
   assert o[4]==0
   total+=o[0];maximum=max(maximum,o[2])
 ar=[json.loads(l) for l in a.with_name(a.name.replace('-logits.f32','-rows.jsonl')).read_text().splitlines()]
 br=[json.loads(l) for l in b.with_name(b.name.replace('-logits.f32','-rows.jsonl')).read_text().splitlines()]
 assert len(ar)==len(br)==rows
 arg=[i for i,(x,y) in enumerate(zip(ar,br)) if x['argmax']!=y['argmax']]
 return {'rows':rows,'max_abs':maximum,'rms':math.sqrt(total/(rows*VOCAB)),'bit_identical':first is None,'first_numeric_difference':first,'numeric_difference_count':len(different),'first_argmax_difference':arg[0] if arg else None,'argmax_difference_count':len(arg),'argmax_differences':arg,'sha256':sha(b),'reference_sha256':sha(a)}

def logits(root,width,suffix=''):
 return root/f'{MODE}-w{width}{suffix}-logits.f32'

if __name__=='__main__':
 D=R/'boundary-output';ref=logits(D/'legacy',1)
 result={'legacy':{},'general':{},'controls':{},'traces':{}}
 for width in [1,3,4]:result['legacy'][str(width)]=compare(ref,logits(D/'legacy',width))
 for width in [1,2,3,4]:result['general'][str(width)]=compare(ref,logits(D/'general',width))
 result['controls']['trace']=compare(logits(D/'legacy',4),logits(D/'legacy-no-trace',4))
 old_explain='08c5eeabd45ac1e4f4adeac6073b3f06a4028e524c200e28e19940c1acd3db19'
 result['controls']['explain256']={'sha256':sha(ref,256*VOCAB*4),'expected_sha256':old_explain}
 for width in [1,4]:
  p=logits(D/'baseline',width)
  result['controls'][f'baseline95-w{width}']={'sha256':sha(p),'expected_sha256':'188c8690abade05294ac66643c56bc7727b21adb3ad6a65b5061e6ca0dfaed5b'}
 for variant,widths in [('legacy',[1,3,4]),('general',[1,2,3,4])]:
  for width in widths:
   rows=[json.loads(l) for l in (D/variant/f'{MODE}-w{width}-trace.jsonl').read_text().splitlines()]
   assert len(rows)==1023*3
   points=[r for r in rows if r['absolute_position'] in [253,254,255,256,257,509,510,511,512,513,514,765,766,767,768,769,770,1021,1022,1023,1024,1025,1026]]
   result['traces'][f'{variant}-w{width}']={'rows':len(rows),'raw_limits':sorted(set(r['raw_limit'] for r in rows)),'boundaries':points}
 assert result['controls']['trace']['bit_identical']
 assert all(v['sha256']==v['expected_sha256'] for k,v in result['controls'].items() if k!='trace')
 (R/'boundary-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='traces'},indent=2))

 S=R/'schedule-output'
 if S.exists():
  result={'histories':{},'controls':{},'attempts':{},'long_prompt':{}}
  cases=[('scalar-rs3',1,'-rs3'),('mixed-append',4,'-schedule1-rs0'),('mixed-rollback',4,'-schedule2-rs3')]
  for variant,width,suffix in cases:
   result['histories'][variant]=compare(ref,logits(S/variant,width,suffix))
   attempts=[json.loads(l) for l in (S/variant/f'{MODE}-w{width}{suffix}-attempts.jsonl').read_text().splitlines()]
   start=0
   for a in attempts:
    assert a['start']==start and a['width']-a['keep']==a['removed'] and 1<=a['keep']<=a['width']<=4
    start+=a['keep']
   assert start==1023
   result['attempts'][variant]={'attempts':len(attempts),'widths':sorted(set(a['width'] for a in attempts)),'rollback_events':sum(a['removed']>0 for a in attempts),'rejected_tokens':sum(a['removed'] for a in attempts),'maximum_rollback':max(a['removed'] for a in attempts),'accepted_tokens':start,'rollback_attempts':[a for a in attempts if a['removed']]}
  lp=logits(S/'long-prompt',1,'-rs0')
  result['long_prompt']['w4']=compare(lp,logits(S/'long-prompt',4,'-rs0'))
  result['controls']['long-prompt-first95']={'sha256':sha(lp,95*VOCAB*4),'expected_sha256':'188c8690abade05294ac66643c56bc7727b21adb3ad6a65b5061e6ca0dfaed5b'}
  assert result['controls']['long-prompt-first95']['sha256']==result['controls']['long-prompt-first95']['expected_sha256']
  (R/'schedule-summary.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))
