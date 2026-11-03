from pathlib import Path
import json,re,collections
P=Path(__file__).parent;root=P/'payload-176128';es=[json.loads(l)for l in(root/'audit.jsonl').open()];frames={x['id']:x for x in es if x['event']=='frame'};ends={x['id']:x for x in es if x['event']=='frame_end'};assert frames.keys()==ends.keys();matches=[];unmatched=0
for i,f in frames.items():assert f['flag']=='1'and ends[i]['status']==0 and not ends[i]['changed_metadata']
for e in es:
 if e['event']!='copy':continue
 assert e['return']==0
 f=frames[e['id']];ts=f['tensors'];rows=[x['ne'][1]for x in ts if x['name']=='idx_score-20'];assert len(set(rows))==1,rows
 q=rows[0];out=[]
 if q>6 and e['api']=='cudaMemcpyPeerAsync'and e['src_device']==3 and e['dst_device']==4:
  n=next(t['ne'][0]for t in ts if t['name']=='idx_score-20')
  for si in e['source_match']:
   a=ts[si]
   if 'candidate_' in a['name']:
    assert e['bytes']<n*q*4,('unexpected wide dense candidate transfer',a['name'],e['bytes'],n,q)
 for si in e['source_match']:
  a=ts[si]
  for di in e['destination_match']:
   b=ts[di];category=None
   if a['name']=='idx_score-20'and re.fullmatch(r'CUDA3#idx_score-20#\d+',b['name']):category='score20';sd,dd=2,3
   elif a['name']=='candidate_block_score-20'and re.fullmatch(r'CUDA3#candidate_block_score-20#\d+',b['name']):category='pool_scores';sd,dd=2,3
   elif a['name']=='candidate_mask-20'and re.fullmatch(r'CUDA4#candidate_mask-20#\d+',b['name']):category='dense_mask';sd,dd=3,4
   elif a['op']=='RESHAPE'and a['type']=='f32'and a['ne'][0]==1 and a['ne'][2]==q and a['root_name'].startswith('node_') and re.fullmatch(r'CUDA4#'+re.escape(a['name'])+r'#\d+',b['name']):category='compact_keep';sd,dd=3,4
   if category is None or e['api']!='cudaMemcpyPeerAsync'or e['src_device']!=sd or e['dst_device']!=dd:continue
   assert e['src']==a['data']and e['dst']==b['data']and e['bytes']==a['bytes']==b['bytes']and a['ne']==b['ne']and a['nb']==b['nb']and a['type']==b['type']=='f32'
   assert a['buffer']==f'CUDA{sd}'and b['buffer']==f'CUDA{dd}'and b['op']=='NONE'and b['flags']&3==3
   consumers=[c for c in f['consumers']if c['tensor']==b['tensor']];assert consumers
   if category=='compact_keep':assert any(c['op']=='REPEAT'and c['node'].startswith('candidate_local_repeat-')for c in consumers)
   out.append({'category':category,'queries':q,'frame':e['id'],'bytes':e['bytes'],'src':a,'dst':b,'consumers':consumers,'copy':e})
 if out:
  assert len({x['category']for x in out})==1;matches.append(out[0])
 else:unmatched+=1
def stats(items):
 return{cat:{'calls':len(rr),'logical_api_bytes':sum(x['bytes']for x in rr),'destination_slots':sorted({int(x['dst']['name'].rsplit('#',1)[1])for x in rr})}for cat in ['score20','pool_scores','dense_mask','compact_keep']for rr in [[x for x in items if x['category']==cat]]}
wide=[m for m in matches if 6<m['queries']<=512];small=[m for m in matches if m['queries']<=6];w=stats(wide);s=stats(small)
assert w['score20']['calls']==w['dense_mask']['calls']==0
assert w['pool_scores']['calls']==w['compact_keep']['calls']>0
assert w['pool_scores']['logical_api_bytes']==w['compact_keep']['logical_api_bytes']
expected=sum(any(t['name']=='idx_score-20'and 6<t['ne'][1]<=512 and t['ne'][0]>16384 for t in f['tensors'])for f in frames.values());assert w['compact_keep']['calls']==expected
assert s['compact_keep']['calls']==s['pool_scores']['calls']==0 and s['score20']['calls']==s['dense_mask']['calls']>0
j={'complete':True,'scope':'Read-only metadata/CUDA copy API proof, full src+dst pointers/ne/nb/type/ranges, expected devices, protected per-slot clone, actual REPEAT consumer, stable metadata. API requests, not physical DMA or end-to-end timing.','frames':len(frames),'expected_wide_frames':expected,'wide':w,'small_fallback':s,'unmatched':unmatched,'matches':matches};(P/'audit-analysis.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({k:v for k,v in j.items()if k!='matches'},indent=2))
