import hashlib,json,statistics
from pathlib import Path

R=Path(__file__).resolve().parent
result={'scope':'Isolated layer0 down/weight/sum only; same frozen inputs as Stage15; no target-model or HTTP throughput estimate. Timings exclude input copies and callback duplication.','architectures':{}}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arch in ['ada','ampere']:
    d=R/f'down-cost-{arch}'
    rows=[json.loads(l) for l in (d/'results.jsonl').read_text().splitlines()]
    times=[json.loads(l) for l in (d/'timings.jsonl').read_text().splitlines()]
    assert len(rows)==44 and all(r['repeat_stable'] for r in rows)
    native=[r for r in rows if r['label'].startswith('native-')]
    assert len(native)==22
    for r in native:
        old=R.parent/f'stage15/down0-{arch}'/(r['label'].removeprefix('native-')+'.f32')
        assert sha(old)==sha(d/(r['label']+'.f32'))
    scalar=[r for r in rows if r['label'].startswith('scalar-down-')]
    assert all(r['max_vs_scalar']==0 and r['column_max']==0 for r in scalar)
    medians={}
    for w in [1,2,4]:
        m={mode:statistics.median(r['mean_ms'] for r in times if r['mode']==mode and r['width']==w) for mode in ['native','scalar-down']}
        m['ratio_scalar_over_native']=m['scalar-down']/m['native'];medians[str(w)]=m
    result['architectures'][arch]={'cases':len(rows),'native_sha_controls_exact':len(native),
        'scalar_width_controls_exact':sum(not r['actual'] for r in scalar),
        'timings_median_ms':medians,'timing_groups':5,'iterations_per_group':100}
(R/'down-cost-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
