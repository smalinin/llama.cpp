#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
D=ROOT/'n1-server'
results=json.loads((D/'results.json').read_text())
assert len(results)==2
responses={}
summary={'scope':'Four prompts, one warmup and three measured repeats per mode, plus two diagnostic baseline requests',
         'snapshot':'stage8/candidate-bin','configurations':{},'comparisons':{}}
for result in results:
    assert result['status']=='passed' and result['server_exit_code']==0
    assert len(result['requests'])==18
    config=result['config'];responses[config]={}
    out={'prompts':{},'peak_gpu_used_mib':result['peak_gpu_used_mib'],
         'peak_process_rss_kib':result['peak_process_rss_kib']}
    for p in ['baseline','svg','python','explain']:
        records=[r for r in result['requests'] if r['phase']=='measure' and r['prompt']==p]
        items=[json.loads((D/config/(r['request']+'-response.json')).read_text()) for r in records]
        assert len(items)==3 and all(x['tokens']==items[0]['tokens'] and x['content']==items[0]['content'] for x in items)
        assert all(x['timings']['cache_n']==0 for x in items)
        responses[config][p]=items[0]
        out['prompts'][p]={'output_length':len(items[0]['tokens']),'stop_type':items[0]['stop_type'],
                           'stable_tokens_and_text':True,
                           'median_tps':statistics.median(x['timings']['predicted_per_second'] for x in items),
                           'min_tps':min(x['timings']['predicted_per_second'] for x in items),
                           'max_tps':max(x['timings']['predicted_per_second'] for x in items),
                           'draft_n':items[0]['timings'].get('draft_n',0),
                           'accepted':items[0]['timings'].get('draft_n_accepted',0),
                           'tokens_sha256':hashlib.sha256(json.dumps(items[0]['tokens']).encode()).hexdigest()}
    summary['configurations'][config]=out
for p in responses['off']:
    a,b=responses['off'][p],responses['n1-p0'][p]
    first=next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None)
    if first is None and len(a['tokens'])!=len(b['tokens']):first=min(len(a['tokens']),len(b['tokens']))
    summary['comparisons'][p]={'same_tokens':a['tokens']==b['tokens'],'same_text':a['content']==b['content'],
                               'first_difference_zero_based':first,
                               'speed_change_percent':100*(summary['configurations']['n1-p0']['prompts'][p]['median_tps']/
                                                        summary['configurations']['off']['prompts'][p]['median_tps']-1)}
    for repeat in range(1,4):
        name=f'{p}-measure-{repeat}-request.json'
        assert json.loads((D/'off'/name).read_text())==json.loads((D/'n1-p0'/name).read_text())
(ROOT/'n1-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
