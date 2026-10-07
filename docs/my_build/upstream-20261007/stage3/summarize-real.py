#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path('/home/sergei/_my_sync/llama_upstream_review')
S=ROOT/'stage3'
def read(p):return json.loads(p.read_text())
def diff(a,b):
    mismatch=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    if mismatch is None and len(a)!=len(b):mismatch=min(len(a),len(b))
    return {'equal':a==b,'first_difference_zero_based':mismatch,'before_length':len(a),'after_length':len(b)}
results=[]
for name in ['glm5next','qwen4exp','glm-dsa','deepseek41']:
    directory=S/'runs'/f'{name}-spec-1'
    if not (directory/'result.json').exists():continue
    before=ROOT/'stage2'/('fixed-after' if name=='glm5next' else 'runs')/f'{name}-spec-1'
    result={'profile':name,'real_result':read(directory/'result.json'),'standard':[],'state_requests':[]}
    for label in ['greedy-1','greedy-2','sampling-1','sampling-2']:
        if not (directory/f'{label}-response.json').exists():continue
        a=read(before/f'{label}-response.json');b=read(directory/f'{label}-response.json')
        result['standard'].append({'request':label,'request_parameters_equal':read(before/f'{label}-request.json')==read(directory/f'{label}-request.json'),
                                  'tokens':diff(a['tokens'],b['tokens']),'text_equal':a['content']==b['content'],
                                  'draft_equal':{k:a['timings'].get(k)==b['timings'].get(k) for k in ['draft_n','draft_n_accepted']},
                                  'before_timings':a['timings'],'after_timings':b['timings']})
    for label in ['restored','truncated','old-version']:
        path=directory/f'state-{label}-response.json'
        if not path.exists():continue
        b=read(path);references={}
        for reference in ['greedy-1','greedy-2']:
            a=read(directory/f'{reference}-response.json');references[reference]=diff(a['tokens'],b['tokens'])
        result['state_requests'].append({'request':label,'references':references,'timings':b.get('timings'),'tokens_predicted':b.get('tokens_predicted')})
    results.append(result)
if (S/'runs-q8/qwen4exp-spec-1/result.json').exists():
    directory=S/'runs-q8/qwen4exp-spec-1'
    result={'profile':'qwen4exp-q8-kv','real_result':read(directory/'result.json'),'state_requests':[]}
    for label in ['restored','truncated','old-version']:
        path=directory/f'state-{label}-response.json'
        if path.exists():
            b=read(path);result['state_requests'].append({'request':label,'greedy_reference':diff(read(directory/'greedy-1-response.json')['tokens'],b['tokens']),'timings':b.get('timings')})
    results.append(result)
(S/'real-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
for result in results:
    print(result['profile'],result['real_result']['status'],sum(r['tokens']['equal'] for r in result.get('standard',[])), '/',len(result.get('standard',[])))
    for r in result['state_requests']:print(' ',r['request'],r.get('references',r.get('greedy_reference')))
