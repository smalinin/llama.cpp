#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
from array import array
import math
R=Path(__file__).resolve().parent;D=R/'model-output'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def compare(reference,actual):
 assert reference.stat().st_size==actual.stat().st_size
 expected=sha(reference);observed=sha(actual)
 if expected==observed:return {'bit_identical':True,'max_abs':0.0,'rms':0.0,'first_float_difference':None,'reference_sha256':expected,'actual_sha256':observed}
 maximum=total=0.0;first=None;count=offset=0
 with reference.open('rb') as rf,actual.open('rb') as af:
  while True:
   rb=rf.read(1024*1024);ab=af.read(1024*1024)
   if not rb:break
   x=array('f');x.frombytes(rb);y=array('f');y.frombytes(ab)
   count+=len(x)
   if rb!=ab:
    bitsx=array('I');bitsx.frombytes(rb);bitsy=array('I');bitsy.frombytes(ab)
    for i,(u,v) in enumerate(zip(x,y)):
     if first is None and bitsx[i]!=bitsy[i]:first=offset+i
     d=float(u)-float(v);maximum=max(maximum,abs(d));total+=d*d
   offset+=len(x)
 return {'bit_identical':False,'max_abs':maximum,'rms':math.sqrt(total/count),'first_float_difference':first,'reference_sha256':expected,'actual_sha256':observed}

parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
out={'cached':{},'controls':{},'raw_layout':{}}
for name in ['dense','ratio1','ratio2']:
 ref=D/f'cache-{name}-w1/logits.f32'
 if not (D/f'cache-{name}-w1/counts.json').exists():
  if args.partial:continue
  raise RuntimeError('missing '+str(ref))
 for width in [1,2,3,4]:
  directory=D/f'cache-{name}-w{width}'
  if not (directory/'counts.json').exists():
   if args.partial:continue
   raise RuntimeError('missing '+str(directory))
  c=read(directory/'counts.json');assert (directory/'logits.f32').stat().st_size==c['rows']*c['vocab']*4
  out['cached'][f'{name}-w{width}']=compare(ref,directory/'logits.f32')
  rows=[json.loads(l) for l in (directory/'rows.jsonl').read_text().splitlines()]
  out['cached'][f'{name}-w{width}']['argmax_reference_identical']=all(r['argmax']==r['reference'] for r in rows)
  trace=[json.loads(l) for l in (directory/'raw-trace.jsonl').read_text().splitlines()];raw=[r for r in trace if r['layer']==0]
  out['raw_layout'][f'{name}-w{width}']={'source_extents':sorted({r['source_raw'] for r in raw}),'effective_extents':sorted({r['raw'] for r in raw}),'crops':sum(r['raw']<r['source_raw'] for r in raw),'first_position':raw[0]['pos'],'last_position':raw[-1]['pos'],'transitions':[r for i,r in enumerate(raw) if not i or r['raw']!=raw[i-1]['raw']],'restored_used_max':read(directory/'restored-cells.json')['used_max']}
  out['raw_layout'][f'{name}-w{width}']['physical_trace_identical_to_scalar']=True
  if width!=1:
   scalar=[json.loads(l) for l in (D/f'cache-{name}-w1/raw-trace.jsonl').read_text().splitlines()]
   keyed={(r['pos'],r['layer']):r for r in scalar}
   out['raw_layout'][f'{name}-w{width}']['physical_trace_identical_to_scalar']=all(r['raw']==keyed[(r['pos'],r['layer'])]['raw'] and r['physical_index']==keyed[(r['pos'],r['layer'])]['physical_index'] and r['effective_total']==keyed[(r['pos'],r['layer'])]['effective_total'] for r in trace)
controls={'baseline-w1':'188c8690abade05294ac66643c56bc7727b21adb3ad6a65b5061e6ca0dfaed5b','baseline-w4':'188c8690abade05294ac66643c56bc7727b21adb3ad6a65b5061e6ca0dfaed5b','sparse-control-w4':'be6d665f41cb69b996ec0fa11c891b8074eebd74987a32ffff541003ebf068d9','history-control-w4':'68c2fb2ca976ff655ff3c8dc7b3f4b72ceaf09c768831c801a4a02500f24ee41'}
for name,expected in controls.items():
 if not (D/name/'counts.json').exists():
  if args.partial:continue
  raise RuntimeError('missing '+name)
 actual=sha(D/name/'logits.f32');out['controls'][name]={'sha256':actual,'expected_sha256':expected,'bit_identical':actual==expected}
out['all_bit_identical']=all(r['bit_identical'] for group in ['cached','controls'] for r in out[group].values())
(R/('model-partial-summary.json' if args.partial else 'model-summary.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'cached':out['cached'],'controls':out['controls'],'all_bit_identical':out['all_bit_identical']},indent=2))
