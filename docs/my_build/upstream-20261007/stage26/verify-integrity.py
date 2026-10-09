#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp')
SNAP=R.parent/'stage8/candidate-bin'
def read(p):return json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
libs=read(R.parent/'stage8/candidate-binary-sha256.json')
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
selection=read(R/'source-selection.json');assert all(sha(REPO/n)==h for n,h in selection['files'].items())
head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();assert head==selection['head']
for name in ['general-callback.h','raw-layout.h']:assert sha(R/name)==sha(R.parent/'stage24'/name)
for prefix,binary,cases,output,number in [('replay','cache-replay','cases.txt','model-output',10),('verify','verify-replay','verify-cases.txt','verify-output',6)]:
    build=read(R/f'{prefix}-build-manifest.json');run=read(R/f'{prefix}-run-manifest.json')
    assert build['exit_code']==run['exit_code']==0 and run['head']==head and not run['production_change']
    assert all(sha(R/n)==h for n,h in build['sources'].items())
    assert sha(R/binary)==run['binary_sha256'] and sha(R/cases)==run['cases_sha256']
    assert run['snapshot_sha256']==libs and all(sha(p)==h for p,h in run['input_sha256'].items())
    assert len(list((R/output).glob('*/counts.json')))==number
    loaded=(R/output/'loaded-libraries.txt').read_text().splitlines();assert loaded
    assert all(Path(line.split()[-1]).parent==SNAP and sha(Path(line.split()[-1]))==libs[Path(line.split()[-1]).name] for line in loaded)
model=read(R/'model-summary.json');assert model['completed_count']==10 and len(model['histories'])==6
assert model['all_controls_exact'] and model['all_tested_full_logits_exact'] and model['scalar_native_control_passed']
assert len(model['native_argmax'])==8 and all(x['all_native_ids_matched'] for x in model['native_argmax'].values())
assert all(x['exact'] for x in model['physical_traces'].values())
b=read(R/'build-manifest.json');assert b['exit_code']==0 and b['head']==head and not b['production_change']
assert all(sha(R/n)==h for n,h in b['sources'].items())
assert all(sha(p)==h for p,h in b['reused_build_inputs'].items())
assert all(sha(R/'experiment-bin'/n)==h for n,h in b['experiment_hashes'].items())
assert [n for n,h in libs.items() if b['experiment_hashes'][n]!=h]==['libllama-server-impl.so']
assert sha(REPO/'tools/server/server-context.cpp')==b['original_server_source_sha256']
d=R/'server-runs/candidate-n3';m=read(d/'manifest.json');capture=read(d/'result.json')
assert capture['status']=='passed' and capture['server_exit_code']==0 and len(capture['requests'])==2
assert m['binary_hashes']==b['experiment_hashes'] and m['loaded_libraries']
for p,h in m['loaded_libraries'].items():assert Path(p).parent==R/'experiment-bin' and sha(p)==h==b['experiment_hashes'][Path(p).name]
for item in capture['requests']:
    label=item['request'];request=read(d/f'{label}-request.json');response=read(d/f'{label}-response.json')
    prior=read(R.parent/'stage25/free-runs/candidate-n3'/f'{label}-response.json')
    assert request==read(R.parent/'stage25/free-runs/candidate-n3'/f'{label}-request.json')
    assert all(response.get(k)==prior.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated'])
    assert len(request['prompt'])==6617 and request['ignore_eos'] and request['n_predict']==len(response['tokens'])==1024
    assert item['tokens_sha256']==hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
    assert item['content_sha256']==hashlib.sha256(response['content'].encode()).hexdigest()
schedules=read(R/'schedule-inputs.json')
assert all(all(x[k] for k in ['response_tokens_content_stop_exact','generation_schedule_exact','physical_layout_exact']) for x in schedules['capture_controls'].values())
assert all(sha(R/'schedules'/n)==x['sha256'] for n,x in schedules['schedules'].items())
verify=read(R/'verify-summary.json');assert verify['completed_count']==6 and len(verify['cases'])==4
assert all(x['bit_identical'] for x in verify['controls'].values())
for kind,position,first in [('fresh',7020,6910),('cache',7401,7251)]:
    actual=verify['cases'][f'{kind}-actual'];native=verify['cases'][f'{kind}-native']
    assert actual['server_argmax']['different_rows']==0
    assert actual['server_full_logits']['rows']==actual['server_full_logits']['exact_rows']
    for c in [actual,native]:
        assert c['first_scalar_difference']['pos']==first and c['first_native_argmax_difference']['pos']==position
        layout=c['first_native_prefix_layout_difference'];assert layout['pos']==first
        assert layout['actual']['physical_index']==0 and layout['scalar']['physical_index']==766
        assert c['physical_layout_checks_passed']
    cross=verify['actual_vs_native_correct_prefix'][kind];assert cross['metadata_different_rows']==0 and cross['saved_full_logit_rows']==cross['saved_full_logit_exact_rows']
for p in R.iterdir():
    if p.suffix in ['.py','.h'] or p.name in ['README.md','cache-replay.cpp','verify-replay.cpp']:assert p.read_bytes().isascii(),p
for p in R.glob('*.patch'):assert all(line.isascii() for line in p.read_text().splitlines() if line.startswith('+'))
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
for name,stat in read(R.parent/'stage7/model-inspection.json')['models'].items():
    st=Path(stat['path']).stat();assert st.st_size==stat['size'] and st.st_mtime_ns==stat['mtime_ns']
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt','model-partial-summary.json','model-partial-analysis.txt','verify-partial-summary.json','verify-partial-analysis.txt'}
skipdirs={'__pycache__','experiment-bin','server-build'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not skipdirs&set(p.relative_to(R).parts) and p.name not in skip and p.suffix not in ['.a','.o']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':head,'snapshot_hashes_verified':32,'stage24_header_and_layout_unchanged':True,'model_replays':16,
     'fixed_history_replays':10,'recorded_schedule_replays':6,'server_processes':1,'successful_completions':2,
     'fixed_wide_scalar_full_logit_comparisons_exact':6,'native1024_argmax_controls_exact':8,'prior95_full_logit_controls_exact':4,
     'server_capture_vs_stage25_exact':2,'captured_generation_events':[571,567],
     'target_only_vs_server_argmax_rows_exact':[904,1764],'target_only_vs_server_full_logit_rows_exact':[22,55],
     'first_schedule_logit_differences':[6910,7251],'first_schedule_token_differences':[7020,7401],
     'native_future_replacements_still_diverge':True,'first_differences_coincide_with_physical_ring_remap':True,
     'internal_operator_cause_localized':False,'production_correctness_or_speed_fix':False,
     'gpu_memory_after_mib':[15]*6,'model_and_draft_stat_verified':True,'production_source_changes':[],
     'installed_server_changed':False,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
     'scope':'Single-slot diagnostic reproduction of two Stage25 long N3 failures. Physical-layout/rollback dependence is reproduced; first affected operator, arbitrary cache states and production CUDA throughput remain open.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
