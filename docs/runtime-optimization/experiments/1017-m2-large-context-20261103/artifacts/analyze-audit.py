from pathlib import Path
import json,collections
P=Path(__file__).parent;p=P/'payload-176128/transfers.jsonl';rows=[json.loads(l)for l in p.read_text().splitlines()];frames={x['id']:x for x in rows if x['event']=='frame'};ends={x['id']:x for x in rows if x['event']=='frame_end'};copies=[x for x in rows if x['event']=='copy'];pairs=[];groups={};overview=[]
for fid,f in frames.items():
 c=[x for x in copies if x['id']==fid];d=collections.defaultdict(list)
 for x in c:d[(x['api'],x.get('src_device'),x.get('dst_device'),x.get('kind'),x['src'],x['bytes'],x['stream'])].append(x)
 for key,g in d.items():
  if len(g)<2 or len({x['dst']for x in g})<2:continue
  sm=sorted({i for x in g for i in x['source_match']});t=[f['tensors'][i]for i in sm];names=[x['name']for x in t];roots={x['root']for x in t};data={x['data']for x in t};rootdata={x['root_data']for x in t};has_base=any('(reshaped)'not in n for n in names);has_view=any('(reshaped)'in n for n in names)
  pair={'frame':fid,'api':key[0],'src_device':key[1],'dst_device':key[2],'src':key[4],'bytes':key[5],'stream':key[6],'calls':len(g),'destinations':[x['dst']for x in g],'source_metadata':t,'source_alias_exact':len(roots)==len(data)==len(rootdata)==1 and has_base and has_view and all(x['contiguous']and x['bytes']==key[5]and x['view_offset']==0 for x in t),'sequence':[x['seq']for x in g],'api_return':[x['return']for x in g]};pair['destination_metadata']=[[z for z in f['tensors']if z['data']==x['dst']and z['bytes']==x['bytes']]for x in g];pair['destination_consumers']=[[z for z in f['consumers']if any(z['tensor']==t['tensor']for t in dm)]for dm in pair['destination_metadata']];pair['source_alias_exact'] = pair['source_alias_exact']and all(pair['destination_metadata'])and all(pair['destination_consumers']);pairs.append(pair)
 overview.append({'frame':fid,'calls':len(c),'end':ends.get(fid),'selected_names':[x['name']for x in f['tensors']]})
assert frames and set(frames)==set(ends)and all(x['status']==0 and not x['changed_metadata']for x in ends.values()),'incomplete/changed frames'
for p in pairs:
 if p['source_alias_exact']:
  n='l_last-14'if any('l_last-14'in t['name']for t in p['source_metadata'])else'l_last-6';k=(n,p['src_device'],p['dst_device'],p['bytes']);g=groups.setdefault(k,{'name':n,'src_device':p['src_device'],'dst_device':p['dst_device'],'bytes_per_request':p['bytes'],'frame_count':0,'api_requests':0,'potential_redundant_api_bytes':0});g['frame_count']+=1;g['api_requests']+=p['calls'];g['potential_redundant_api_bytes']+=(p['calls']-1)*p['bytes']
out={'complete':True,'scope':'Read-only API scheduling/metadata audit for one8K+16 request after two capacity requests.No CUDA activity trace/payload read;not physical DMA proof or speed A/B.','frames':len(frames),'copy_api_calls_overlapping_selected_ranges':len(copies),'repeated_source_groups':len(pairs),'exact_alias_groups':sum(x['source_alias_exact']for x in pairs),'groups':list(groups.values()),'potential_redundant_api_bytes':sum(x['potential_redundant_api_bytes']for x in groups.values()),'pairs':pairs,'frame_overview':overview,'first_frame':next(iter(frames.values()))}
(P/'audit-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k not in['pairs','frame_overview','first_frame']})
