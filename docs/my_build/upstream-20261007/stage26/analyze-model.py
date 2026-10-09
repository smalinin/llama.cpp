#!/usr/bin/env python3
from pathlib import Path
from array import array
import argparse,ctypes,hashlib,json,math,runpy

R=Path(__file__).resolve().parent
D=R/'model-output'
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
metric=runpy.run_path(str(R.parent/'stage16/analyze-chain.py'))['metric']
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def compare(a,b):
    ca,cb=read(a/'counts.json'),read(b/'counts.json');assert ca['rows']==cb['rows'] and ca['vocab']==cb['vocab']
    n,vocab=ca['rows'],ca['vocab'];sa,sb=sha(a/'logits.f32'),sha(b/'logits.f32')
    first=None;count=0;maximum=total=0.0
    assert all((p/'logits.f32').stat().st_size==n*vocab*4 for p in [a,b])
    if sa!=sb:
        with (a/'logits.f32').open('rb') as af,(b/'logits.f32').open('rb') as bf:
            for row in range(n):
                ab,bb=af.read(vocab*4),bf.read(vocab*4)
                if ab==bb:continue
                count+=1;x=array('f');x.frombytes(ab);y=array('f');y.frombytes(bb)
                out=(ctypes.c_double*5)();metric(x.buffer_info()[0],y.buffer_info()[0],vocab,0,out);assert out[4]==0
                maximum=max(maximum,out[2]);total+=out[0]
                if first is None:first={'output_index':row,'query_position':ca['prompt_tokens']+row-1,'max_abs':out[2],'rms':math.sqrt(out[0]/vocab)}
    ar,br=lines(a/'rows.jsonl'),lines(b/'rows.jsonl');assert len(ar)==len(br)==n
    return {'bit_identical':sa==sb,'reference_sha256':sa,'actual_sha256':sb,'rows':n,'first_difference':first,
            'different_rows':count,'max_abs':maximum,'rms':math.sqrt(total/(n*vocab)),
            'argmax_different_rows':[x['index'] for x,y in zip(ar,br) if x['argmax_no_eog']!=y['argmax_no_eog']]}

result={'controls':{},'histories':{},'counts':{},'indexer_selection':{},'physical_traces':{},'native_argmax':{}}
completed={p.parent.name for p in D.glob('*/counts.json')}
layers=[2,8,14,20,24,28,32,36]
for label in sorted(completed):
    c=read(D/label/'counts.json');result['counts'][label]=c
    counts=read(D/label/'indexer-counts.json');w=c['width'];q=c['rows']-1
    expected=sum(min(w,q-start) for start in range(0,q,w) if min(w,q-start)>1)
    required={f'blk.{layer}.indexer.proj.weight':expected for layer in layers} if expected else {}
    assert counts==required,(label,counts,required);result['indexer_selection'][label]={'exact':True,'columns_per_projection':expected,'counts':counts}
    if '-native-' in label:
        rows=lines(D/label/'rows.jsonl');different=[x for x in rows if x['argmax_no_eog']!=x['reference']]
        result['native_argmax'][label]={'matches':c['rows']-len(different),'rows':c['rows'],'all_native_ids_matched':not different,
                                      'first_difference':different[0] if different else None,'different_indexes':[x['index'] for x in different]}
for label in ['baseline-w1','baseline-w4']:
    if label in completed:
        actual=sha(D/label/'logits.f32');reference=sha(R.parent/'stage24/model-output'/label/'logits.f32')
        result['controls'][label]={'actual_sha256':actual,'reference_sha256':reference,'bit_identical':actual==reference}
for kind in ['fresh','cache']:
    scalar=f'{kind}-native-w1'
    if scalar not in completed:continue
    trace={(x['pos'],x['layer']):x for x in lines(D/scalar/'raw-trace.jsonl')}
    for w in [2,3,4]:
        label=f'{kind}-native-w{w}'
        if label not in completed:continue
        result['histories'][label]=compare(D/scalar,D/label)
        rows=lines(D/label/'raw-trace.jsonl')
        result['physical_traces'][label]={'rows':len(rows),'exact':len(rows)==len(trace) and all(all(x[k]==trace[(x['pos'],x['layer'])][k] for k in ['raw','physical_index','effective_total','n_kv_max']) for x in rows),'raw_extents':sorted({x['raw'] for x in rows if x['layer']==0})}
result['completed_count']=len(completed);result['completed_cases']=sorted(completed)
result['all_controls_exact']=all(v['bit_identical'] for v in result['controls'].values())
result['scalar_native_control_passed']=all(result['native_argmax'][f'{kind}-native-w1']['all_native_ids_matched'] for kind in ['fresh','cache'] if f'{kind}-native-w1' in completed)
result['all_tested_full_logits_exact']=all(v['bit_identical'] for v in result['histories'].values())
result['scope']='Target-only forced replay of exact Stage25 native histories with unchanged Stage24 callback; no draft, speculative rollback or free generation.'
if not args.partial:assert len(completed)==10 and len(result['controls'])==2 and len(result['histories'])==6
(R/('model-partial-summary.json' if args.partial else 'model-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['counts','indexer_selection']},indent=2))
