#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp')
SNAP=R.parent/'stage8/candidate-bin'
CONFIGS=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']
def read(p):return json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

libs=read(R.parent/'stage8/candidate-binary-sha256.json')
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
assert all(sha(REPO/n)==h for n,h in read(R/'source-selection.json')['files'].items())
for name in ['general-callback.h','raw-layout.h']:
    assert sha(R/name)==sha(R.parent/'stage24'/name)
b=read(R/'build-manifest.json')
assert b['exit_code']==0 and all(sha(R/n)==h for n,h in b['sources'].items())
assert all(sha(p)==h for p,h in b['reused_build_inputs'].items())
assert all(sha(R/'experiment-bin'/n)==h for n,h in b['experiment_hashes'].items())
assert [n for n,h in libs.items() if b['experiment_hashes'][n]!=h]==['libllama-server-impl.so']
assert sha(REPO/'tools/server/server-context.cpp')==b['original_server_source_sha256']
assert sha(R.parent/'stage24/general-callback.h')==b['parent_header_sha256']
summary=read(R/'free-summary.json')
assert summary['completed_configs']==CONFIGS and summary['comparisons']==64
assert len(summary['probability_output_controls'])==20 and len(summary['fresh_vs_cache'])==20
assert len(summary['native_prior_controls'])==14 and len(summary['cache_reuse'])==20
assert not summary['request_errors']
old=R.parent/'stage21';native=old/'free-runs/snapshot-off';index=read(old/'raw-file-sha256.json')
for p in native.glob('*.json'):
    assert sha(p)==index[str(p.relative_to(old))]['sha256']
for config in CONFIGS:
    d=R/'free-runs'/config;m=read(d/'manifest.json');result=read(d/'result.json')
    assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==16
    expected=libs if config=='snapshot-off' else b['experiment_hashes']
    assert m['binary_hashes']==expected and m['loaded_libraries']
    assert m['parent_header_sha256']==sha(R/'general-callback.h')
    for p,h in m['loaded_libraries'].items():
        assert Path(p).parent==Path(m['library_directory']) and expected[Path(p).name]==h and sha(p)==h
    for item in result['requests']:
        label=item['request'];request=read(d/f'{label}-request.json');response=read(d/f'{label}-response.json')
        assert item['prompt_sha256']==hashlib.sha256(json.dumps(request['prompt']).encode()).hexdigest()
        assert item['tokens_sha256']==hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
        assert item['content_sha256']==hashlib.sha256(response['content'].encode()).hexdigest()
        assert len(response['tokens'])==response['tokens_predicted']
        if '-long-' in label:
            assert len(request['prompt'])==6617 and request['ignore_eos'] and request['n_predict']==len(response['tokens'])==1024
        else:
            assert request==read(native/f'{label}-request.json')
    full=read(d/'long-tokenize-response.json')['tokens']
    for name,count in [('dense',3033),('ratio1',3289),('ratio2',6617)]:
        prompt=read(d/f'{name}-prompt.json');assert len(prompt)==count and prompt==full[:count-128]+full[-128:]
for config in ['candidate-n1','candidate-n3']:
    for name,count in [('dense',3033),('ratio1',3289),('ratio2',6617),('ratio2-long',6617)]:
        for state in ['fresh','cache']:
            label=f'{config}/{name}-{state}';e=summary['requests'][label];a=summary['acceptance'][label]
            last={'dense':3072,'ratio1':3328,'ratio2':6657,'ratio2-long':7639}[name]
            assert e['decode_min']==count and e['decode_max']>=last
            assert e['attention']>0 and e['indexer']>0 and e['physical_plan_checked_each_decode'] and e['prefill_interventions_zero']
            assert a['draft_n']>0 and 0<=a['draft_n_accepted']<=a['draft_n']
source=(R/'server-context-general.cpp').read_text()
assert 'params_dft.cb_eval = nullptr;' in source and 'params_dft.cb_eval_user_data = nullptr;' in source
assert 'ds17_down.inputs.clear();' in source
assert 'for (const auto & item : ds14_saved_precision) ggml_mul_mat_set_prec(item.first, item.second);' in source
for p in R.iterdir():
    if p.suffix in ['.py','.h'] or p.name=='README.md':assert p.read_bytes().isascii(),p
for p in R.glob('*.patch'):
    assert all(line.isascii() for line in p.read_text().splitlines() if line.startswith('+'))
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
for name,stat in read(R.parent/'stage7/model-inspection.json')['models'].items():
    st=Path(stat['path']).stat();assert st.st_size==stat['size'] and st.st_mtime_ns==stat['mtime_ns']
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt','free-partial-summary.json','free-partial-analysis.txt'}
skipdirs={'__pycache__','experiment-bin','server-build'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not skipdirs&set(p.relative_to(R).parts) and p.name not in skip and p.suffix not in ['.a','.o']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
equal=lambda v:all(v[k] for k in ['tokens_identical','content_identical','stop_metadata_identical'])
out={'head':b['head'],'snapshot_hashes_verified':32,'experimental_builds':1,'modified_libraries_per_build':['libllama-server-impl.so'],
     'stage24_header_unchanged':True,'stage24_raw_layout_unchanged':True,'server_processes':5,'successful_completions':80,
     'native_prior_controls_exact':sum(equal(v) for v in summary['native_prior_controls'].values()),
     'integration_off_vs_native_exact':sum(equal(v) for k,v in summary['against_native'].items() if k.startswith('integration-off/')),
     'candidate_off_vs_native_exact':sum(equal(v) for k,v in summary['against_native'].items() if k.startswith('candidate-off/')),
     'speculative_vs_native_exact':sum(equal(v) for k,v in summary['against_native'].items() if k.startswith(('candidate-n1/','candidate-n3/'))),
     'long1024_completions':10,'long1024_speculative_vs_native_exact':sum(equal(v) for k,v in summary['against_native'].items() if k.startswith(('candidate-n1/','candidate-n3/')) and '-long-' in k),
     'n_probs5_0_pairs_exact':sum(equal(v) for v in summary['probability_output_controls'].values()),
     'fresh_cache_pairs_exact':sum(equal(v) for v in summary['fresh_vs_cache'].values()),
     'cache_requests_reusing_prompt_minus_four':20,'overall_server_compatibility_passed':summary['overall_server_compatibility_passed'],
     'matching_native_criterion_passed':summary['matching_native_criterion_passed'],'cache_independence_criterion_passed':summary['cache_independence_criterion_passed'],
     'speculative_differences':{k:v['first_difference'] for k,v in summary['against_native'].items() if k.startswith(('candidate-n1/','candidate-n3/')) and not equal(v)},
     'native_long_fresh_cache_first_difference':summary['fresh_vs_cache']['snapshot-off/ratio2-long']['first_difference'],
     'http_errors':0,'gpu_memory_after_mib':[15]*6,'model_and_draft_stat_verified':True,'production_source_changes':[],
     'installed_server_changed':False,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
     'scope':'Bounded single-slot free greedy generation with Stage24 diagnostic callback; arbitrary cache states, full-session restore, multi-slot execution and production CUDA throughput remain untested.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
