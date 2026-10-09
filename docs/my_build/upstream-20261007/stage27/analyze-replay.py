#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parent;D=R/'model-output';OLD=R.parent/'stage26'
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
completed={p.parent.name for p in D.glob('*/counts.json')}
out={'completed_count':len(completed),'controls':{},'schedules':{}}
for label in ['baseline-w1','baseline-w4','fresh-native-w1','cache-native-w1']:
    if label not in completed:continue
    actual=sha(D/label/'logits.f32');reference=sha(OLD/'model-output'/label/'logits.f32')
    rows=lines(D/label/'rows.jsonl');key='argmax_no_eog' if '-native-' in label else 'argmax';native=[r for r in rows if r[key]!=r['reference']]
    out['controls'][label]={'full_logits_exact':actual==reference,'actual_sha256':actual,'reference_sha256':reference,
                            'rows':len(rows),'native_argmax_different_rows':len(native)}
for kind in ['fresh','cache']:
    label=f'{kind}-actual'
    if label not in completed:continue
    counts=read(D/label/'counts.json');rows=lines(D/label/'rows.jsonl');assert len(rows)==counts['rows']
    good=[r for r in rows if r['native_prefix']]
    wrong_logits=[r for r in good if not r['scalar_exact']];wrong_ids=[r for r in good if r['argmax_no_eog']!=r['reference']]
    scalar={(r['pos'],r['layer']):r for r in lines(OLD/'model-output'/f'{kind}-native-w1/raw-trace.jsonl')}
    trace=lines(D/label/'raw-trace.jsonl');wrong_layout=[]
    for t in trace:
        s=scalar[(t['pos'],t['layer'])]
        if any(t[k]!=s[k] for k in ['physical_index','raw','effective_total','n_kv_max']):wrong_layout.append({'actual':t,'scalar':s})
    schedule=[list(map(int,s.split())) for s in (OLD/'schedules'/f'{kind}-actual-prefix.txt').read_text().splitlines()]
    total=sum(e[1] for e in schedule);wide=sum(e[1] for e in schedule if e[1]>1)
    assert len(rows)==total+1 and len(trace)==total*3
    assert read(D/label/'indexer-counts.json')=={f'blk.{layer}.indexer.proj.weight':wide for layer in [2,8,14,20,24,28,32,36]}
    first_pos=6910 if kind=='fresh' else 7251;token_pos=7020 if kind=='fresh' else 7401
    out['schedules'][kind]={'events':len(schedule),'rows':len(rows),'native_prefix_rows':len(good),
                           'scalar_exact_rows':len(good)-len(wrong_logits),'first_logit_difference':wrong_logits[0] if wrong_logits else None,
                           'native_argmax_different_rows':len(wrong_ids),'first_argmax_difference':wrong_ids[0] if wrong_ids else None,
                           'physical_trace_rows':len(trace),'physical_different_rows':len(wrong_layout),'first_physical_difference':wrong_layout[0] if wrong_layout else None,
                           'indexer_columns_per_source':wide,
                           'former_numerical_boundary':[r for r in good if r['pos']==first_pos],
                           'former_token_boundary':[r for r in good if r['pos']==token_pos],
                           'former_boundary_layout':[r for r in trace if r['layer']==0 and r['pos']==first_pos and r['col']==0]}
if not args.partial:assert len(completed)==6 and len(out['controls'])==4 and len(out['schedules'])==2
out['all_completed_controls_exact']=all(x['full_logits_exact'] for x in out['controls'].values())
out['all_completed_prefix_logits_exact']=all(x['first_logit_difference'] is None for x in out['schedules'].values())
out['scope']='Same recorded actual tokens and rollback positions as Stage26; only libllama tail head selection changes. Callback and raw-plan arithmetic remain byte-identical.'
(R/('replay-partial-summary.json' if args.partial else 'replay-summary.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
