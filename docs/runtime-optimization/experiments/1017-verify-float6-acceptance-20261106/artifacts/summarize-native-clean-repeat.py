from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
assert P.parent.name == '1017-verify-float6-acceptance-20261106'
load = lambda name: json.loads((P / name).read_text())
sha = lambda name: hashlib.sha256((P / name).read_bytes()).hexdigest()
d = load('native-clean-repeat-results.json')
assert d['complete'] and d['clean_reproduction_only']
assert len(d['arms']) == 1
a = d['arms'][0]
assert a['label'] == 'clean-on-repeat' and a['flag'] == '1'
assert a['complete'] and a['exit_code'] == 0 and a['interposer_absent']
assert not a['floor_breach'] and not a['warnings'] and not a.get('forced_kill')
assert 'LD_PRELOAD' not in a['environment']
assert [(r['prefix'], r['kind']) for r in a['requests']] == [(512,'warmup'), (512,'measured'), (512,'measured'), (8192,'warmup')]
old = load('native-results.json')
assert not old['complete']
old_off = old['arms'][0]
old_on = old['arms'][1]
for arm in [old_off, old_on]:
    assert arm['command'][:-1] == a['command'][:-1]
    assert {k:v for k,v in arm['environment'].items() if k != 'LLAMA_DSV41_VERIFY_FLOAT_WIDE'} == {k:v for k,v in a['environment'].items() if k != 'LLAMA_DSV41_VERIFY_FLOAT_WIDE'}
rows = []
for i, r in enumerate(a['requests']):
    v = r['response']; t = v['timings']
    assert t['cache_n'] == 0 and t['prompt_n'] == r['prefix']
    assert t['predicted_n'] == len(v['tokens']) == 256
    row = dict(prefix=r['prefix'], kind=r['kind'], tenth_token=v['tokens'][9], first16_tokens=v['tokens'][:16], text_start=v['content'][:160], draft_generated=t['draft_n'], draft_accepted=t['draft_n_accepted'], tg=r['tg'], pp=r['pp'])
    for name, arm in [('original_off', old_off), ('failed_on', old_on)]:
        ref = arm['requests'][i]; rv = ref['response']; rt = rv['timings']
        assert ref['prefix'] == r['prefix'] and ref['kind'] == r['kind']
        assert rv['generation_settings'] == v['generation_settings']
        diff = next((j for j,(x,y) in enumerate(zip(v['tokens'],rv['tokens'])) if x != y), None)
        row[name] = dict(first_diff_index=diff, all256_tokens_exact=v['tokens']==rv['tokens'], draft_counts_exact=(t['draft_n'],t['draft_n_accepted'])==(rt['draft_n'],rt['draft_n_accepted']), tenth_token=rv['tokens'][9])
        assert r['first_diff_'+('original_off' if name=='original_off' else 'failed_on')] == diff
    rows.append(row)
floor = min(x['min_free']/2**20 for phase in a['memory'].values() for x in phase.values() if isinstance(x,dict))
assert floor >= 512
out = dict(complete=True,clean_reproduction_only=True,head=d['head'], original_acceptance_remains_failed=True,default_changed=False, runtime_changed=False,interposer_absent=True,all4_match_original_off=all(r['original_off']['all256_tokens_exact'] and r['original_off']['draft_counts_exact'] for r in rows), repeat_8K_first9_match_original_off=a['requests'][-1]['response']['tokens'][:9]==old_off['requests'][3]['response']['tokens'][:9],first_8K_difference_from_original_off=rows[-1]['original_off']['first_diff_index'],actual_min_free_mib=floor,exit_code=a['exit_code'],rows=rows,hashes={f:sha(f) for f in ['native-results.json','native-clean-repeat-results.json','native-clean-repeat.py','native-clean-repeat-protocol.json']},scope='One fresh clean ON process and original first4 request history, no interposer/callback/profiler. Not a new bracketing OFF/ON/OFF acceptance or speed series; keep original failed manifest and defaults unchanged.')
(P / 'native-clean-repeat-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
