from pathlib import Path
import hashlib
import json
import statistics

P = Path(__file__).resolve().parent
assert P.parent.name == '1017-verify-float6-acceptance-20261106'
src = P / 'native-results.json'
d = json.loads(src.read_text())
assert d['complete'] and d['all_outputs_counters_exact']
assert [a['label'] for a in d['arms']] == ['off-pre', 'on', 'off-post']
refs = {}
floors = {}
for a in d['arms']:
    assert a['complete'] and a['exit_code'] == 0
    assert not a['floor_breach'] and not a['warnings'] and not a.get('forced_kill')
    assert len(a['requests']) == 9
    assert all('/1017-verify-float6-20261106/artifacts/candidate/' in m for m in a['maps'])
    values = [v['min_free'] / 2**20 for phase in ['load', 'inference'] for v in a['memory'][phase].values() if isinstance(v, dict)]
    floors[a['label']] = min(values)
    assert floors[a['label']] >= 512
    for r in a['requests']:
        v = r['response']
        t = v['timings']
        assert t['cache_n'] == 0 and t['prompt_n'] == r['prefix']
        assert t['predicted_n'] == len(v['tokens']) == 256
        h = hashlib.sha256(json.dumps(v['tokens'], separators=(',', ':')).encode()).hexdigest()
        assert h == r['tokens_sha256']
        value = (v['tokens'], t['draft_n'], t['draft_n_accepted'])
        if r['prefix'] in refs:
            assert refs[r['prefix']] == value
        else:
            refs[r['prefix']] = value
rows = []
for prefix in [512, 8192, 90112]:
    arms = {}
    for a in d['arms']:
        rr = [r for r in a['requests'] if r['prefix'] == prefix and r['kind'] == 'measured']
        assert len(rr) == 2
        arms[a['label']] = {metric: [r[metric] for r in rr] for metric in ['pp', 'tg']}
    row = {'prefix': prefix, 'arms': arms}
    for metric in ['pp', 'tg']:
        off = arms['off-pre'][metric] + arms['off-post'][metric]
        on = arms['on'][metric]
        baseline = statistics.mean(off)
        candidate = statistics.mean(on)
        row[metric] = dict(off_mean=baseline, on_mean=candidate,
                           off_range=[min(off), max(off)], on_range=[min(on), max(on)],
                           off_pre_mean=statistics.mean(arms['off-pre'][metric]),
                           off_post_mean=statistics.mean(arms['off-post'][metric]),
                           off_post_vs_pre_pct=100*(statistics.mean(arms['off-post'][metric])/statistics.mean(arms['off-pre'][metric])-1),
                           delta_pct=100*(candidate/baseline-1))
    _, generated, accepted = refs[prefix]
    row['draft'] = dict(proposed=generated, accepted=accepted, acceptance_pct=100*accepted/generated)
    rows.append(row)
out = dict(complete=True, raw_manifest_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
           all_27_output_arrays_hashes_and_draft_counts_exact=True, measured_requests_per_arm=6,
           measured_requests_per_prefix=dict(off=4, on=2), warmups_excluded_from_speed=True,
           actual_min_free_mib=floors, actual_global_min_free_mib=min(floors.values()), rows=rows,
           method='Arithmetic mean of two ON measurements versus four bracketing OFF measurements per prefix. Native TG = 255000/predicted_ms for 256 generated tokens; HTTP predicted_per_second uses 256. No confidence interval or universal determinism claim. All warmup outputs included in exactness check.')
(P / 'native-summary.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
