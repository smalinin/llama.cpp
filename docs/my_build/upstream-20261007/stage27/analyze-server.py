#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent;D=R/'server-runs/candidate-n3';OLD=R.parent/'stage25/free-runs'
def read(p):return json.loads(p.read_text())
def first(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return {'output_index':i,'reference_token':x,'actual_token':y}
    return None if len(a)==len(b) else {'output_index':min(len(a),len(b)),'length_difference':True}
result=read(D/'result.json');assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==2
out={'requests':{},'completions':2,'http_errors':0}
log=(D/'server.log').read_bytes()
for item in result['requests']:
    label=item['request'];response=read(D/f'{label}-response.json');reference=read(OLD/'snapshot-off'/f'{label}-response.json');prior=read(OLD/'candidate-n3'/f'{label}-response.json')
    assert read(D/f'{label}-request.json')==read(OLD/'snapshot-off'/f'{label}-request.json')
    assert len(response['tokens'])==response['tokens_predicted']==1024
    controls={k:response.get(k)==reference.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated']}
    events=[{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',m.group(1))} for m in re.finditer(r'DS14_EVENT (.*)',log[item['log_start_offset']:item['log_end_offset']].decode())]
    assert events
    out['requests'][label]={'native_comparison':controls,'native_first_difference':first(reference['tokens'],response['tokens']),
                            'prior_candidate_first_difference':first(prior['tokens'],response['tokens']),
                            'cache_n':response['timings'].get('cache_n',0),'prompt_n':response['timings']['prompt_n'],
                            'draft_n':response['timings'].get('draft_n'),'draft_n_accepted':response['timings'].get('draft_n_accepted'),
                            'generation_batches':sum(bool(e['generation']) for e in events),
                            'response_tokens_sha256':hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()}
    for e in events:
        assert e['ret']==0 and e['enabled']==1
        if not e['generation'] or e['width']==1:assert e['indexer']==e['indexer_sources']==0
        else:assert e['indexer_sources']==8 and e['indexer']==8*e['width']
out['all_two_native_answers_exact']=all(all(r['native_comparison'].values()) for r in out['requests'].values())
out['scope']='Two long N3 fresh/cache requests with Stage24 diagnostic arithmetic and the production rollback fix. This does not establish general production N3 logit equivalence or speedup.'
(R/'server-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
