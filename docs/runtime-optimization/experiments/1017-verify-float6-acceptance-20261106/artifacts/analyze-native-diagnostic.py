from pathlib import Path
import array
import hashlib
import json

P = Path(__file__).resolve().parent
assert P.parent.name == '1017-verify-float6-acceptance-20261106'
def read(name): return json.loads((P/name).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def signature(r):
    t=r['response']['timings']
    return [r['response']['tokens'],t['draft_n'],t['draft_n_accepted']]
d=read('native-trace-results.json')
assert d['complete'] and d['diagnostic_capture_only']
assert [a['label'] for a in d['arms']]==['trace-on','trace-off']
streams=[]
floors={}
for a in d['arms']:
    assert a['complete'] and a['exit_code']==0 and len(a['requests'])==4
    assert not a['warnings'] and not a['floor_breach']
    floors[a['label']]=min(v['min_free']/2**20 for phase in a['memory'].values() for v in phase.values() if isinstance(v,dict))
    assert floors[a['label']]>=512
    root=P/('native-'+a['label'])
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    logits=[e for e in events if e['kind']=='logits']
    assert logits and logits[0]['prediction_pos']==8192
    for e in logits:
        assert sha(root/e['raw'])==e['sha256'] and e['nonfinite']==0
    streams.append((root,events,logits))
rows=[]
assert len(streams[0][2])==len(streams[1][2])
for a,b in zip(streams[0][2],streams[1][2]):
    assert (a['prediction_pos'],a['idx'])==(b['prediction_pos'],b['idx'])
    raw_a=(streams[0][0]/a['raw']).read_bytes();raw_b=(streams[1][0]/b['raw']).read_bytes()
    va=array.array('f');va.frombytes(raw_a);vb=array.array('f');vb.frombytes(raw_b)
    assert len(va)==len(vb)==a['vocab']
    rows.append(dict(prediction_pos=a['prediction_pos'],raw_equal=raw_a==raw_b,
        differing_float_bits=sum(raw_a[i:i+4]!=raw_b[i:i+4] for i in range(0,len(raw_a),4)),
        max_abs=max(abs(x-y) for x,y in zip(va,vb)),on_top=a['top'][:2],off_top=b['top'][:2]))
normal=lambda es:[{k:v for k,v in e.items() if k!='id'} for e in es if e['kind']=='decode']
on=normal(streams[0][1]);off=normal(streams[1][1])
first_decode_difference=next((i for i,(a,b) in enumerate(zip(on,off)) if a!=b),None)
original=read('native-results.json')
a=next(r for r in original['arms'][0]['requests'] if r['prefix']==8192)
b=original['arms'][1]['requests'][-1]
first_mismatch=next(i for i,(x,y) in enumerate(zip(a['response']['tokens'],b['response']['tokens'])) if x!=y)
control=read('native-control-results.json');assert control['complete'] and control['all_outputs_counters_exact']
out=dict(diagnostic_capture_only=True,complete=True,native_acceptance_pass=False,
    original_manifest_sha256=sha(P/'native-results.json'),trace_manifest_sha256=sha(P/'native-trace-results.json'),
    control_manifest_sha256=sha(P/'native-control-results.json'),
    original_first_mismatch_index=first_mismatch,original_off_token=a['response']['tokens'][first_mismatch],original_on_token=b['response']['tokens'][first_mismatch],
    diagnostic_response_pairs_exact=all(signature(a)==signature(b) for a,b in zip(d['arms'][0]['requests'],d['arms'][1]['requests'])),
    diagnostic_matches_original_off=all(r['matches_original_off'] for a in d['arms'] for r in a['requests']),
    decode_event_counts=[len(on),len(off)],decode_events_exact=on==off,first_decode_difference=first_decode_difference,
    logit_pairs=len(rows),all_captured_logits_exact=all(r['raw_equal'] for r in rows),actual_min_free_mib=floors,rows=rows,
    conclusion='Diagnostic instrumentation did not reproduce the clean ON failure. This does not repair or supersede failed native acceptance, prove the mechanism correct, or identify the root cause. No diagnostic timing is accepted as speed evidence.')
(P/'native-diagnostic-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
