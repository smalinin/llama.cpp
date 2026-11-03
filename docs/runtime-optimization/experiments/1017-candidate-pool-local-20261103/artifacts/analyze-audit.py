from pathlib import Path
import json,re,collections
P=Path(__file__).parent;root=P/'payload-176128';es=[json.loads(l)for l in(root/'audit.jsonl').open()];frames={e['id']:e for e in es if e['event']=='frame'};ends={e['id']:e for e in es if e['event']=='frame_end'};assert frames.keys()==ends.keys();matches=[];excluded=0
for i,f in frames.items():assert f['flag']=='1' and not ends[i]['changed_metadata']and ends[i]['status']==0
for e in es:
 if e['event']!='copy':continue
 assert e['return']==0
 f=frames[e['id']];ts=f['tensors'];out=[]
 for si in e['source_match']:
  a=ts[si]
  for di in e['destination_match']:
   b=ts[di]
   for cat,name,sd,dd in [('score20','idx_score-20',2,3),('blocks20','candidate_block_score-20',2,3),('mask20','candidate_mask-20',3,4)]:
    if a['name']!=name or not re.fullmatch(rf'CUDA{dd}#{name}#\d+',b['name']):continue
    if e['api']!='cudaMemcpyPeerAsync'or e['src_device']!=sd or e['dst_device']!=dd:continue
    assert e['src']==a['data']and e['dst']==b['data']and e['bytes']==a['bytes']==b['bytes']and a['ne']==b['ne']and a['nb']==b['nb']and a['type']==b['type']=='f32'
    assert a['buffer']==f'CUDA{sd}'and b['buffer']==f'CUDA{dd}'and b['op']=='NONE'and b['flags']&3==3
    consumers=[c for c in f['consumers']if c['tensor']==b['tensor']];assert consumers
    out.append({'category':cat,'frame':e['id'],'bytes':e['bytes'],'src':a,'dst':b,'consumers':consumers,'copy':e})
 if out:
  assert len({x['category']for x in out})==1;matches.append(out[0])
 else:excluded+=1
def stats(items):
 return {cat:{'calls':len(rr),'logical_api_bytes':sum(m['bytes']for m in rr),'destination_slots':sorted(set(int(m['dst']['name'].rsplit('#',1)[1])for m in rr))}for cat in ['score20','blocks20','mask20']for rr in [[m for m in items if m['category']==cat]]}
wide=[m for m in matches if 6<m['src']['ne'][1]<=512];fallback=[m for m in matches if m['src']['ne'][1]<=6];summary=stats(wide);fallback_summary=stats(fallback)
assert summary['score20']['calls']==0 and summary['blocks20']['calls']>0 and summary['blocks20']['calls']==summary['mask20']['calls'];assert summary['mask20']['logical_api_bytes']==8*summary['blocks20']['logical_api_bytes']
assert fallback_summary['blocks20']['calls']==0 and fallback_summary['score20']['calls']==fallback_summary['mask20']['calls']>0
assert all(m['src']['ne'][0]>2048 and m['src']['ne'][1]>6 for m in wide if m['category']=='blocks20')
assert all(any(c['node']=='candidate_pinned_block_score-20'and c['op']=='ADD'for c in m['consumers'])for m in wide if m['category']=='blocks20')
expected=sum(any(t['name']=='candidate_block_score-20'and t['ne'][1]>6 for t in f['tensors'])for f in frames.values());assert summary['blocks20']['calls']==expected
out={'complete':True,'scope':'Read-only full-pointer/full-size metadata + CUDA API request proof. Protected destination clone, expected src/dst CUDA ordinal, shape/stride/type, actual consumer, stable metadata. Logical requests only, not hardware DMA counters or speed attribution. Partial/address-only coincidences excluded.','frames':len(frames),'expected_candidate_frames':expected,'wide_summary':summary,'fallback_summary':fallback_summary,'fallback_note':'The raw probe also admits reshaped keep tensors whose ne1 is n_blocks; therefore generation3/4 queries are captured. Classification uses canonical source query width. Tiny fallback copies remain intentionally unchanged.','excluded_unrelated_or_unmatched_copies':excluded,'matches':matches};(P/'audit-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='matches'},indent=2))
