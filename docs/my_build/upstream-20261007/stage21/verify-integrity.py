#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
libs=read(R.parent/'stage8/candidate-binary-sha256.json');summary=read(R/'free-summary.json')
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
builds={}
for name,directory in [('build-manifest.json','experiment-bin'),('build-extent-manifest.json','extent-experiment-bin')]:
 b=read(R/name);builds[name]=b
 assert b['exit_code']==0 and all(sha(R/n)==h for n,h in b['sources'].items())
 assert all(sha(p)==h for p,h in b['reused_build_inputs'].items())
 assert all(sha(R/directory/n)==h for n,h in b['experiment_hashes'].items())
 assert [n for n,h in libs.items() if b['experiment_hashes'][n]!=h]==['libllama-server-impl.so']
 assert sha(R/'general-callback.h')==sha(R.parent/'stage19/general-callback.h')==sha(R.parent/'stage20/general-callback.h')==b['reused_header_sha256']
 assert sha(REPO/'tools/server/server-context.cpp')==b['original_server_source_sha256']
for selection in ['source-selection.json','cache-source-selection.json']:
 assert all(sha(REPO/n)==h for n,h in read(R/selection)['files'].items())
configs=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n1-remaining','candidate-n3']
assert summary['completed_configs']==configs and summary['comparisons']==64
assert all(summary['checks'].values()) and not summary['overall_server_compatibility_passed']
assert len(summary['extent_failures'])==2
for config,width in [('candidate-n1-remaining',2),('candidate-n3',4)]:
 e=summary['extent_failures'][config]
 assert e['pos']==6617 and e['width']==width and e['raw']==e['total']==256 and e['raw_limit']==768 and e['compressed']==0
assert len(summary['request_errors'])==3 and len(summary['cache_reuse'])==9
assert all(v['cache_n']>0 and v['prompt_n']==4 for v in summary['cache_reuse'].values())
assert len(summary['probability_output_controls'])==20 and len(summary['fresh_vs_cache'])==9
successful=0
for config in configs:
 d=R/'free-runs'/config;m=read(d/'manifest.json');result=read(d/'result.json')
 b=builds['build-extent-manifest.json' if config in ['candidate-n1-remaining','candidate-n3'] else 'build-manifest.json']
 assert m['head']==b['head'] and result['server_exit_code']==0 and m['draft_callback'] is False
 expected={'snapshot-off':14,'integration-off':14,'candidate-off':14,'candidate-n1':9,'candidate-n1-remaining':2,'candidate-n3':11}
 assert len(result['requests'])==expected[config];successful+=len(result['requests'])
 assert result['status']==('passed' if config in configs[:3] else 'failed')
 hashes=libs if config=='snapshot-off' else b['experiment_hashes'];assert m['binary_hashes']==hashes and m['loaded_libraries']
 for p,h in m['loaded_libraries'].items():
  assert Path(p).parent==Path(m['library_directory']) and hashes[Path(p).name]==h and sha(p)==h
 for item in result['requests']:
  request=read(d/(item['request']+'-request.json'))
  assert item['prompt_sha256']==hashlib.sha256(json.dumps(request['prompt']).encode()).hexdigest()
  response=read(d/(item['request']+'-response.json'))
  assert item['content_sha256']==hashlib.sha256(response['content'].encode()).hexdigest()
  assert item['tokens_sha256']==hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
 for name,count in [('dense',3033),('ratio1',3289),('ratio2',6617)]:
  full=read(d/'long-tokenize-response.json')['tokens'];prompt=read(d/(name+'-prompt.json'))
  assert len(prompt)==count and prompt==full[:count-128]+full[-128:]
 for p in d.glob('*-response.json'):
  response=read(p)
  if 'tokens_predicted' in response:assert len(response['tokens'])==response['tokens_predicted']
 if config in configs[3:]:
  error=read(d/'request-failure.json');assert error['http_status']==500
  assert 'unsupported cache extent' in error.get('body',error.get('server_message',''))
for config in ['candidate-n1','candidate-n3']:
 for p in ['dense','ratio1','ratio2']:
  actual='candidate-n1-remaining' if config=='candidate-n1' and p!='dense' else config
  e=summary['requests'][f'{actual}/{p}-fresh']
  assert e['decode_min']=={'dense':3033,'ratio1':3289,'ratio2':6617}[p]
  assert e['decode_max']>={'dense':3072,'ratio1':3328,'ratio2':6657}[p] and e['attention']>0
  if p=='dense':assert e['r1_dense']>0 and e['r1_sparse']==0 and e['r2_sparse']==0
  if p=='ratio1':assert e['r1_dense']>0 and e['r1_sparse']>0 and e['r2_sparse']==0
  if p=='ratio2':assert e['r2_dense']>0 and e['r2_sparse']>0
for name in ['server-context-general.cpp','server-context-extent.cpp']:
 source=(R/name).read_text()
 assert 'params_dft.cb_eval = nullptr;' in source and 'params_dft.cb_eval_user_data = nullptr;' in source
 assert 'for (const auto & item : ds14_saved_precision) ggml_mul_mat_set_prec(item.first, item.second);' in source
 assert 'ds17_down.inputs.clear();' in source and 'ds14_options.decoding = ds14_enabled && ds14_generating;' in source
for p in R.iterdir():
 if p.suffix in ['.py','.h'] or p.name=='README.md':assert p.read_bytes().isascii(),p
for p in R.glob('*.patch'):assert all(line.isascii() for line in p.read_text().splitlines() if line.startswith('+'))
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
model=read(R.parent/'stage7/model-inspection.json')['models']['model'];st=Path(model['path']).stat();assert st.st_size==model['size'] and st.st_mtime_ns==model['mtime_ns']
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt'};skipdirs={'__pycache__','experiment-bin','server-build','extent-experiment-bin','extent-server-build'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not skipdirs&set(p.relative_to(R).parts) and p.name not in skip and p.suffix not in ['.a','.o']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':builds['build-manifest.json']['head'],'snapshot_hashes_verified':32,'modified_snapshot_files':[],'experimental_builds':2,'modified_libraries_per_build':['libllama-server-impl.so'],'diagnostic_header_unchanged':True,'server_processes':6,'successful_completions':successful,'http500_cache_errors':3,'planned_cache_requests_not_run_after_same_guard_failure':3,'nontrivial_native_comparisons_exact':50,'speculative_fresh_vs_native_exact':22,'integration_off_vs_native_exact':14,'candidate_off_vs_native_exact':14,'n_probs5_0_pairs_exact':20,'prior_native_controls_exact':4,'cache_requests_with_positive_reuse':9,'fresh_cache_pairs_exact':9,'successful_decode_events_ret_zero':True,'successful_prefill_interventions_zero':True,'gpu_memory_after_mib':[15]*6,'model_first_shard_stat_verified':True,'production_source_changes':[],'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),'overall_server_compatibility_passed':False,'scope':'All completed responses match native. Diagnostic cache-extent guard rejects restored raw layout; fix and cached speculative checks remain before CUDA/throughput.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
