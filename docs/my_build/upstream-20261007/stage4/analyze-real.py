#!/usr/bin/env python3
import json,statistics,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
pairs=[];failures=[]
for before in sorted((R/'real-before').glob('*-spec-*')):
    after=R/'real-after'/before.name
    if not (before/'result.json').exists() or not (after/'result.json').exists():continue
    rb=json.loads((before/'result.json').read_text());ra=json.loads((after/'result.json').read_text())
    if rb['status']!='passed' or ra['status']!='passed':continue
    mb=json.loads((before/'manifest.json').read_text());ma=json.loads((after/'manifest.json').read_text())
    def normalize(cmd):
        out=cmd[1:];out[out.index('--port')+1]='PORT';return out
    placement_equal=mb['layer_placement']==ma['layer_placement'];buffers_equal=mb['model_buffers']==ma['model_buffers']
    same_parameters=normalize(mb['command'])==normalize(ma['command'])
    pair={'case':before.name,'parameters_equal':same_parameters,'placement_equal':placement_equal,'model_buffers_equal':buffers_equal,'boot_id_equal':mb['boot_id']==ma['boot_id'],'responses':[],'performance':[],'memory':{'before_peak_gpu_used_mib':rb['peak_gpu_used_mib'],'after_peak_gpu_used_mib':ra['peak_gpu_used_mib'],'before_peak_rss_kib':rb['peak_process_rss_kib'],'after_peak_rss_kib':ra['peak_process_rss_kib']}}
    if not (same_parameters and placement_equal and buffers_equal and pair['boot_id_equal']):failures.append({'case':before.name,'reason':'comparison conditions differ'})
    labels=[item['request'] for item in rb['requests']]
    assert labels==[item['request'] for item in ra['requests']]
    for label in labels:
        b=json.loads((before/(label+'-response.json')).read_text());a=json.loads((after/(label+'-response.json')).read_text())
        bt=b.get('tokens',[]);at=a.get('tokens',[])
        equal=bt==at;first=next((i for i,(x,y) in enumerate(zip(bt,at)) if x!=y),None)
        if first is None and len(bt)!=len(at):first=min(len(bt),len(at))
        item={'request':label,'tokens_equal':equal,'text_equal':b.get('content')==a.get('content'),'before_tokens':len(bt),'after_tokens':len(at),'first_token_difference':first,'before_timings':b['timings'],'after_timings':a['timings'],'draft_counts_equal':{k:b['timings'].get(k) for k in ('draft_n','draft_n_accepted')}=={k:a['timings'].get(k) for k in ('draft_n','draft_n_accepted')}}
        pair['responses'].append(item)
        if 'greedy' in label and (not equal or not item['text_equal']):failures.append({'case':before.name,'request':label,'reason':'greedy output changed','first_difference':first})
    groups=[('short-greedy',[l for l in labels if l.startswith('greedy-') and l!='greedy-1']),('long-greedy',[l for l in labels if l.startswith('long-greedy-') and l!='long-greedy-1'])]
    for group,selected in groups:
        if not selected:continue
        metrics={}
        for version,dirpath in [('before',before),('after',after)]:
            ts=[json.loads((dirpath/(l+'-response.json')).read_text())['timings'] for l in selected]
            metrics[version]={k:statistics.median(t[k] for t in ts) for k in ('prompt_ms','predicted_ms','prompt_per_second','predicted_per_second')}
            metrics[version]['samples']=ts
            draft=sum(t.get('draft_n',0) for t in ts);accepted=sum(t.get('draft_n_accepted',0) for t in ts)
            metrics[version]['draft_n']=draft;metrics[version]['draft_n_accepted']=accepted;metrics[version]['acceptance_percent']=100*accepted/draft if draft else None
        stats={'group':group,'requests':selected,**metrics,'prefill_speed_change_percent':100*(metrics['after']['prompt_per_second']/metrics['before']['prompt_per_second']-1),'decode_speed_change_percent':100*(metrics['after']['predicted_per_second']/metrics['before']['predicted_per_second']-1)}
        pair['performance'].append(stats)
    pairs.append(pair)
result={'complete_pairs':len(pairs),'paired_responses':sum(len(p['responses']) for p in pairs),'greedy_equal':sum(r['tokens_equal'] and r['text_equal'] for p in pairs for r in p['responses'] if 'greedy' in r['request']),'sampling_equal':sum(r['tokens_equal'] and r['text_equal'] for p in pairs for r in p['responses'] if 'sampling' in r['request']),'failures':failures,'pairs':pairs}
(R/'real-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
for p in pairs:
    for m in p['performance']:print(p['case'],m['group'],'pp',round(m['prefill_speed_change_percent'],2),'tg',round(m['decode_speed_change_percent'],2))
