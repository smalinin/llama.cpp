#!/usr/bin/env python3
from pathlib import Path
from array import array
import argparse,ctypes,hashlib,json,math,runpy

R=Path(__file__).resolve().parent
D=R/'verify-output'
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
metric=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['metric']
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def compare_bytes(a,b):
    if a==b:return {'bit_identical':True,'max_abs':0.0,'rms':0.0}
    x=array('f');x.frombytes(a);y=array('f');y.frombytes(b);assert len(x)==len(y)
    out=(ctypes.c_double*5)();metric(x.buffer_info()[0],y.buffer_info()[0],len(x),0,out);assert out[4]==0
    return {'bit_identical':False,'max_abs':out[2],'rms':math.sqrt(out[0]/len(x))}

completed={p.parent.name for p in D.glob('*/counts.json')}
result={'controls':{},'cases':{},'completed_count':len(completed),'completed_cases':sorted(completed)}
for label in ['baseline-w1','baseline-w4']:
    if label not in completed:continue
    actual=sha(D/label/'logits.f32');reference=sha(R/'model-output'/label/'logits.f32')
    result['controls'][label]={'bit_identical':actual==reference,'actual_sha256':actual,'reference_sha256':reference}
for kind in ['fresh','cache']:
    capture=R/'server-runs/candidate-n3/capture'
    request=next(x for x in read(R/'server-runs/candidate-n3/result.json')['requests'] if x['request'].endswith(kind))
    data=(capture/'events.jsonl').read_bytes()[request['capture_start_offset']:request['capture_end_offset']]
    events=[json.loads(line) for line in data.splitlines()]
    captured={(e['event'],col):r for e in events for col,r in enumerate(e['rows'])}
    for flavour in ['actual','native']:
        label=f'{kind}-{flavour}'
        if label not in completed:continue
        rows=lines(D/label/'rows.jsonl');counts=read(D/label/'counts.json');assert len(rows)==counts['rows']
        schedule=[list(map(int,s.split())) for s in (R/'schedules'/f'{kind}-{flavour}-prefix.txt').read_text().splitlines()]
        expected_rows=1+sum(e[1] for e in schedule);assert len(rows)==expected_rows
        selected=sum(e[1] for e in schedule if e[1]>1)
        expected_indexer={f'blk.{layer}.indexer.proj.weight':selected for layer in [2,8,14,20,24,28,32,36]}
        assert read(D/label/'indexer-counts.json')==expected_indexer
        trace=lines(D/label/'raw-trace.jsonl');assert len(trace)==(expected_rows-1)*3
        good=[r for r in rows if r['native_prefix']]
        numeric=[r for r in good if not r['scalar_exact']]
        argmax=[r for r in good if r['argmax_no_eog']!=r['reference']]
        case={'counts':counts,'generation_events':len(schedule),'native_prefix_rows':len(good),
              'scalar_exact_rows':len(good)-len(numeric),'first_scalar_difference':numeric[0] if numeric else None,
              'scalar_different_rows':len(numeric),'max_abs':max((r['max_abs'] for r in good),default=0.0),
              'first_native_argmax_difference':argmax[0] if argmax else None,'native_argmax_different_rows':len(argmax),
              'indexer_columns_per_projection':selected,'physical_layout_checks_passed':True,'focal_rows':[]}
        if flavour=='actual':
            wrong=[];focal=[]
            with (D/label/'logits.f32').open('rb') as f:
                for r in rows:
                    if r['event']<0:continue
                    key=(r['event'],r['pos']-next(e[0] for e in schedule if e[2]==r['event']))
                    c=captured[key];assert r['pos']==c['pos']
                    if any(r[k]!=c[k] for k in ['argmax','argmax_no_eog']):wrong.append({'replay':r,'capture':c})
                    if c['file'] and r['has_logits']:
                        f.seek(r['logits_offset']);a=f.read(counts['vocab']*4);b=(capture/c['file']).read_bytes()
                        assert len(a)==len(b)==counts['vocab']*4
                        focal.append({'event':r['event'],'pos':r['pos'],'native_prefix':r['native_prefix'],**compare_bytes(a,b)})
            case['server_argmax']={'rows':len(rows)-1,'different_rows':len(wrong),'first_difference':wrong[0] if wrong else None}
            case['server_full_logits']={'rows':len(focal),'exact_rows':sum(r['bit_identical'] for r in focal),
                                       'first_difference':next((r for r in focal if not r['bit_identical']),None),
                                       'max_abs':max((r['max_abs'] for r in focal),default=0.0),'rows_detail':focal}
        scalar_trace={(r['pos'],r['layer']):r for r in lines(R/'model-output'/f'{kind}-native-w1/raw-trace.jsonl')}
        layout={};cursor=0
        for e in schedule:
            chunk=trace[cursor:cursor+e[1]*3];cursor+=e[1]*3
            for r in chunk:layout[(e[2],r['pos'],r['layer'])]=r
        different_layout=[]
        for r in good:
            if r['event']<0:continue
            a=layout[(r['event'],r['pos'],0)];b=scalar_trace[(r['pos'],0)]
            if any(a[k]!=b[k] for k in ['physical_index','raw','effective_total','n_kv_max']):
                different_layout.append({'event':r['event'],'pos':r['pos'],'index':r['index'],
                                         'actual':a,'scalar':b,'scalar_exact':r['scalar_exact']})
        case['first_native_prefix_layout_difference']=different_layout[0] if different_layout else None
        target=7020 if kind=='fresh' else 7401
        case['focal_rows']=[r for r in rows if r['pos']==target and r['native_prefix']]
        result['cases'][label]=case
result['actual_vs_native_correct_prefix']={}
for kind in ['fresh','cache']:
    actual=f'{kind}-actual';native=f'{kind}-native'
    if actual not in completed or native not in completed:continue
    ar=[r for r in lines(D/actual/'rows.jsonl') if r['native_prefix']]
    nr={(r['event'],r['pos']):r for r in lines(D/native/'rows.jsonl')}
    metadata=['argmax','argmax_no_eog','scalar_exact','max_abs','rms']
    wrong=[r for r in ar if any(r[k]!=nr[(r['event'],r['pos'])][k] for k in metadata)]
    exact=count=0
    with (D/actual/'logits.f32').open('rb') as af,(D/native/'logits.f32').open('rb') as nf:
        for a in ar:
            b=nr[(a['event'],a['pos'])]
            if not a['has_logits']:continue
            assert b['has_logits'];size=read(D/actual/'counts.json')['vocab']*4
            af.seek(a['logits_offset']);nf.seek(b['logits_offset']);exact+=af.read(size)==nf.read(size);count+=1
    result['actual_vs_native_correct_prefix'][kind]={'rows':len(ar),'metadata_different_rows':len(wrong),
                                                   'saved_full_logit_rows':count,'saved_full_logit_exact_rows':exact}
if not args.partial:assert len(completed)==6 and len(result['cases'])==4 and len(result['controls'])==2
result['scope']='Recorded verification and rollback schedules replayed without a draft model; native variants replace all recorded batch inputs with native history tokens.'
(R/('verify-partial-summary.json' if args.partial else 'verify-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
for label,c in result['cases'].items():print(label,json.dumps({k:v for k,v in c.items() if k not in ['focal_rows','counts','server_full_logits']}))
