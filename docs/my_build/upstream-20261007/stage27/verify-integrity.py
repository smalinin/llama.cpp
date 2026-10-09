#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp')
SNAP=R.parent/'stage8/candidate-bin'
def read(p):return json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
assert head=='fdc24e5ef6ceefba870c228003ef9d841c0422ff'
libs=read(R.parent/'stage8/candidate-binary-sha256.json')
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
b=read(R/'build-manifest.json');candidate=b['candidate_hashes']
assert b['exit_code']==0 and b['head']==head and b['snapshot_hashes']==libs
assert sha(REPO/'src/llama-kv-cache.cpp')==sha(R/'llama-kv-cache.cpp')==b['source_sha256']
assert all(sha(p)==h for p,h in b['reused_objects'].items())
assert all(sha(R/'candidate-bin'/n)==h for n,h in candidate.items())
assert set(b['changed_files'])=={n for n in libs if libs[n]!=candidate[n]}=={'libllama.so','libllama.so.0','libllama.so.0.4.0'}
assert sha(REPO/'build-glm53/bin/libllama.so.0.4.0')==libs['libllama.so.0.4.0']
for n,h in read(R.parent/'stage26/source-selection.json')['files'].items():
    if n!='src/llama-kv-cache.cpp':assert sha(REPO/n)==h,n
changed=subprocess.check_output(['git','-C',str(REPO),'diff','--name-only','--','.',':(exclude)docs/my_build/upstream-20261007'],text=True).splitlines()
assert changed==['src/llama-kv-cache.cpp'],changed
assert subprocess.check_output(['git','-C',str(REPO),'diff','--','src/llama-kv-cache.cpp'])==(R/'source.patch').read_bytes()
for n in ['general-callback.h','raw-layout.h']:assert sha(R/n)==sha(R.parent/'stage24'/n)
tail=read(R/'tail-summary.json');assert tail['build_exit_code']==0 and sha(R/'tail-rollback.cpp')==tail['source_sha256']
assert tail['runs']['before']['summary']=={'cases':27,'failures':7} and tail['runs']['before']['exit_code']==1
assert tail['runs']['after']['summary']=={'cases':27,'failures':0} and tail['runs']['after']['exit_code']==0
for label,hashes in [('before',libs),('after',candidate)]:assert tail['runs'][label]['libllama_sha256']==hashes['libllama.so.0.4.0']
build=read(R/'replay-build-manifest.json');run=read(R/'replay-run-manifest.json')
assert build['exit_code']==run['exit_code']==0 and run['head']==head and run['production_change']
assert run['source_sha256']==b['source_sha256'] and run['snapshot_sha256']==candidate
assert all(sha(R/n)==h for n,h in build['sources'].items())
assert sha(R/'rollback-replay')==run['binary_sha256'] and sha(R/'cases.txt')==run['cases_sha256']
assert all(sha(p)==h for p,h in run['input_sha256'].items())
loaded=(R/'model-output/loaded-libraries.txt').read_text().splitlines();assert loaded
for line in loaded:
    p=Path(line.split()[-1]);assert p.parent==R/'candidate-bin' and sha(p)==candidate[p.name]
m=read(R/'replay-summary.json');assert m['completed_count']==6 and m['all_completed_controls_exact'] and m['all_completed_prefix_logits_exact']
assert len(m['controls'])==4 and all(v['full_logits_exact'] and not v['native_argmax_different_rows'] for v in m['controls'].values())
for kind,rows in [('fresh',405),('cache',786)]:
    v=m['schedules'][kind]
    assert v['native_prefix_rows']==v['scalar_exact_rows']==rows and v['first_logit_difference'] is None
    assert v['native_argmax_different_rows']==v['physical_different_rows']==0
    assert v['former_boundary_layout'][0]['physical_index']==766
sb=read(R/'server-build-manifest.json');parent=read(R.parent/'stage26/build-manifest.json')['experiment_hashes'];ex=sb['experiment_hashes']
assert sb['head']==head and sb['source_sha256']==b['source_sha256'] and not sb['server_rebuilt'] and sb['libllama_rebuilt']
assert {n for n in ex if ex[n]!=parent[n]}==set(b['changed_files'])
assert all(sha(R/'experiment-bin'/n)==h for n,h in ex.items())
assert ex['libllama-server-impl.so']==sb['parent_server_impl_sha256']
server=R/'server-runs/candidate-n3';sm=read(server/'manifest.json');result=read(server/'result.json')
assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==2
assert sm['binary_hashes']==ex and sm['loaded_libraries']
for p,h in sm['loaded_libraries'].items():assert Path(p).parent==R/'experiment-bin' and sha(p)==h==ex[Path(p).name]
ss=read(R/'server-summary.json');assert ss['all_two_native_answers_exact'] and ss['completions']==2 and not ss['http_errors']
for item in result['requests']:
    label=item['request'];request=read(server/f'{label}-request.json');response=read(server/f'{label}-response.json')
    old=R.parent/'stage25/free-runs/snapshot-off'
    assert request==read(old/f'{label}-request.json')
    prior=read(old/f'{label}-response.json')
    assert all(response.get(k)==prior.get(k) for k in ['tokens','content','stop_type','stopping_word','truncated'])
    assert len(request['prompt'])==6617 and request['ignore_eos'] and len(response['tokens'])==request['n_predict']==1024
    assert item['tokens_sha256']==hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
    assert item['content_sha256']==hashlib.sha256(response['content'].encode()).hexdigest()
for p in R.iterdir():
    if p.suffix in ['.py','.h'] or p.name in ['README.md','rollback-replay.cpp','tail-rollback.cpp']:assert p.read_bytes().isascii(),p
for p in R.glob('*.patch'):assert all(line.isascii() for line in p.read_text().splitlines() if line.startswith('+'))
for name,stat in read(R.parent/'stage7/model-inspection.json')['models'].items():
    st=Path(stat['path']).stat();assert st.st_size==stat['size'] and st.st_mtime_ns==stat['mtime_ns']
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt','replay-partial-summary.json','replay-partial-analysis.txt','repo-README.md'}
skipdirs={'__pycache__','experiment-bin','candidate-bin'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not skipdirs&set(p.relative_to(R).parts) and p.name not in skip and p.suffix not in ['.a','.o']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':head,'snapshot_hashes_verified':32,'only_libllama_rebuilt':True,'stage24_callback_and_layout_unchanged':True,
     'tail_regression_cases':27,'tail_regression_failures_before':7,'tail_regression_failures_after':0,
     'model_replays':6,'full_logit_sha_controls_exact':4,'recorded_rollback_prefix_logit_rows_exact':[405,786],
     'physical_trace_rows_exact':[2712,5292],'former_boundaries_now_scalar_exact':[6910,7251,7020,7401],
     'server_processes':1,'successful_completions':2,'long_n3_native_answers_exact':2,'tokens_per_answer':1024,
     'production_source_changes':['src/llama-kv-cache.cpp'],'production_arithmetic_changed':False,
     'installed_server_changed':False,'model_and_draft_stat_verified':True,'gpu_memory_after_mib':[15]*6,
     'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
     'scope':'Narrow DSV4.1 per-stream raw SWA tail rollback. Model equality uses unchanged diagnostic arithmetic; native fresh/cache, session restore and general production N3 equivalence or speedup remain open.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
