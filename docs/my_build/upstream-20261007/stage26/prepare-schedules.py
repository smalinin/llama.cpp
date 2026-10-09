#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re,struct

R=Path(__file__).resolve().parent
D=R/'server-runs/candidate-n3'
OLD=R.parent/'stage25/free-runs/candidate-n3'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def events(data):return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',m.group(1))} for m in re.finditer(r'DS14_EVENT (.*)',data.decode())]

result=read(D/'result.json');assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==2
log=(D/'server.log').read_bytes();capture=(D/'capture/events.jsonl').read_bytes();oldlog=(OLD/'server.log').read_bytes();olditems={x['request']:x for x in read(OLD/'result.json')['requests']}
(R/'schedules').mkdir(exist_ok=True);out={'capture_controls':{},'schedules':{},'focal_events':{}}
for item in result['requests']:
    label=item['request'];kind='fresh' if label.endswith('fresh') else 'cache';target=7020 if kind=='fresh' else 7401
    response=read(D/f'{label}-response.json');old=read(OLD/f'{label}-response.json')
    assert all(response.get(k)==old.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated'])
    es=[x for x in events(log[item['log_start_offset']:item['log_end_offset']]) if x['generation']]
    olditem=olditems[label];oldes=[x for x in events(oldlog[olditem['log_start_offset']:olditem['log_end_offset']]) if x['generation']]
    metas=[json.loads(l) for l in capture[item['capture_start_offset']:item['capture_end_offset']].splitlines()]
    assert len(metas)==len(es)==len(oldes)
    for m,e,o in zip(metas,es,oldes):
        assert (m['pos'],m['width'])==(e['pos'],e['width'])==(o['pos'],o['width'])
        assert all(e[k]==o[k] for k in ['raw_source','raw_first','raw_last','physical_first','physical_last'])
    native=list(struct.unpack('<1024i',(R/'inputs'/f'{kind}.i32').read_bytes()))
    for flavour,full in [('actual-full',True),('actual-prefix',False),('native-prefix',False)]:
        rows=[]
        for m,e in zip(metas,es):
            if not full and m['pos']>target:break
            tokens=m['tokens'] if flavour.startswith('actual') else native[m['pos']-6617:m['pos']-6617+m['width']]
            assert len(tokens)==m['width']
            values=[m['pos'],m['width'],m['event'],e['raw_first'],e['raw_last'],e['raw_source'],e['physical_first'],e['physical_last'],*tokens]
            rows.append(' '.join(map(str,values)))
        path=R/'schedules'/f'{kind}-{flavour}.txt';path.write_text('\n'.join(rows)+'\n')
        out['schedules'][path.name]={'events':len(rows),'sha256':sha(path),'flavour':flavour,'kind':kind,'stop_query':None if full else target}
    out['capture_controls'][kind]={'response_tokens_content_stop_exact':True,'generation_schedule_exact':True,'physical_layout_exact':True,'events':len(metas),'response_sha256':sha(D/f'{label}-response.json')}
    out['focal_events'][kind]=[{**m,'server_layout':{k:e[k] for k in ['raw_source','raw_first','raw_last','physical_first','physical_last']}} for m,e in zip(metas,es) if m['pos']<=target<m['pos']+m['width']]
(R/'schedule-inputs.json').write_text(json.dumps(out,indent=2)+'\n')
baseline=[l for l in (R/'cases.txt').read_text().splitlines() if l.startswith('baseline-')]
cases=baseline
for kind in ['fresh','cache']:
    for flavour in ['actual','native']:
        cases.append(f'{kind}-{flavour} {R}/inputs/prompt.i32 {R}/inputs/{kind}.i32 4 {int(kind=="cache")} {R}/inputs/fresh.i32 {R}/schedules/{kind}-{flavour}-prefix.txt '+
                     (f'{R}/schedules/fresh-actual-full.txt' if kind=='cache' else '-')+f' {R}/model-output/{kind}-native-w1/logits.f32')
(R/'verify-cases.txt').write_text('\n'.join(cases)+'\n')
runner=(R/'run-replay.py').read_text().replace("R/'cache-replay'","R/'verify-replay'").replace("R/'replay-build-manifest.json'","R/'verify-build-manifest.json'")
runner=runner.replace('line.split()[1:3]+line.split()[5:6]',"[p for p in line.split()[1:3]+line.split()[5:] if p!='-']")
(R/'run-verification.py').write_text(runner)
print(json.dumps(out['capture_controls'],indent=2))
