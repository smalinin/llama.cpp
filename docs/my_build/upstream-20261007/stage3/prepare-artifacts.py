#!/usr/bin/env python3
"""Verify final artifacts, retaining the reproduced DeepSeek limitation."""
import json,subprocess
from pathlib import Path
S=Path('/home/sergei/_my_sync/llama_upstream_review/stage3')
summaries={};known_failures=[]
for group in ['runs','runs-q8','continuation-runs','continuation-runs-q8','continuation-before-deep']:
    results=json.loads((S/group/'summary.json').read_text());assert results
    summaries[group]=results
    hashes=json.loads(((S.parent/'stage2' if group=='continuation-before-deep' else S)/'binary-sha256.json').read_text())
    for r in results:
        expected_failure=group in ['continuation-runs','continuation-before-deep'] and r['case']=='deepseek41-spec-0'
        assert r['status']==('failed' if expected_failure else 'passed'),(group,r)
        if expected_failure:
            assert 'continuation differs after restore' in r['error']
            known_failures.append({'group':group,'case':r['case'],'error':r['error']})
        directory=S/group/r['case'];manifest=json.loads((directory/'manifest.json').read_text())
        assert len(manifest['loaded_libraries'])==2
        assert all(hashes[Path(p).name]==h for p,h in manifest['loaded_libraries'].items())
        if 'continuation' in r and r['case'].endswith('spec-0'):
            c=r['continuation']
            assert c['control_timings']['cache_n']==c['restored_timings']['cache_n']==c['expected_cached_tokens']
            assert c['tokens_equal']==c['text_equal']==(not expected_failure)
            assert c['control_timings']['predicted_n']==c['restored_timings']['predicted_n']==32
proof=json.loads((S/'deep-continuation-before-after.json').read_text())
assert proof['before']['first_difference_zero_based']==proof['after']['first_difference_zero_based']==1
assert all(proof[n]['tokens_equal_before_after'] and proof[n]['text_equal_before_after'] for n in ['greedy-1','continuation-control','continuation-restored'])
subprocess.run(['python3',str(S/'summarize-real.py')],check=True)
completion_names={f'{name}-response.json' for name in ['greedy-1','greedy-2','sampling-1','sampling-2','state-restored','state-truncated','state-old-version','continuation-control','continuation-restored']}
counts={}
for group,results in summaries.items():
    count=0
    for r in results:
        for p in (S/group/r['case']).glob('*response.json'):
            if p.name in completion_names:
                response=json.loads(p.read_text());assert 'error' not in response and response.get('tokens')
                count+=1
    counts[group]=count
assert sum(c for g,c in counts.items() if g!='continuation-before-deep')==52
assert counts['continuation-before-deep']==3
checks={'all_expected_results_verified':True,'all_groups_passed':False,'completion_requests_checked':52,'before_completion_requests_checked':3,
        'completion_requests_by_group':counts,'groups':{g:len(d) for g,d in summaries.items()},
        'library_manifests_checked':sum(map(len,summaries.values())),
        'known_failures':known_failures,'deepseek_failure_reproduced_before':True,
        'continuation_profiles':[{'group':g,'profile':r['case'],**r['continuation']} for g,d in summaries.items() for r in d if 'continuation' in r],
        'real_standard_token_matches':sum(x['tokens']['equal'] for r in json.loads((S/'real-comparison.json').read_text()) for x in r.get('standard',[])),
        'note':'Target-only slot files do not save draft/speculative state. DeepSeek native continuation differs before and after these patches, starting at token index 1.'}
(S/'real-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Final artifacts verified; existing DeepSeek native continuation limitation remains')
