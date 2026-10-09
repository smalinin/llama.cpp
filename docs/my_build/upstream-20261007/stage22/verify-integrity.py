#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import subprocess
import struct

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp')
SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
libs=read(R.parent/'stage8/candidate-binary-sha256.json')
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
b=read(R/'build-manifest.json')
assert b['exit_code']==0 and all(sha(R/n)==h for n,h in b['sources'].items())
assert all(sha(p)==h for p,h in b['reused_build_inputs'].items())
assert all(sha(R/'experiment-bin'/n)==h for n,h in b['experiment_hashes'].items())
assert [n for n,h in libs.items() if b['experiment_hashes'][n]!=h]==['libllama-server-impl.so']
assert sha(REPO/'tools/server/server-context.cpp')==b['original_server_source_sha256']
assert sha(R.parent/'stage21/general-callback.h')==b['parent_header_sha256']
assert all(sha(REPO/n)==h for n,h in read(R/'source-selection.json')['files'].items())
mb=read(R/'replay-build-manifest.json')
assert mb['exit_code']==0 and all(sha(R/n)==h for n,h in mb['sources'].items())
for prefix,out in [('replay','model-output')]:
 m=read(R/(prefix+'-run-manifest.json'))
 assert m['exit_code']==0 and sha(m['command'][0])==m['binary_sha256']
 assert sha(m['command'][2])==m['cases_sha256']
 assert all(sha(p)==h for p,h in m['input_sha256'].items())
 assert m['snapshot_sha256']==libs
 loaded=(R/out/'loaded-libraries.txt').read_text().splitlines()
 assert loaded
 for line in loaded:
  path=Path(line.split()[-1]);assert path.parent==SNAP and sha(path)==libs[path.name]
model=read(R/'model-summary.json')
assert len(model['cached'])==12 and len(model['controls'])==4 and model['all_bit_identical']
assert all(v['argmax_reference_identical'] for v in model['cached'].values())
assert all(v['physical_trace_identical_to_scalar'] for v in model['raw_layout'].values())
long=read(R/'model-long-summary.json')
assert not long['bit_identical'] and long['physical_trace_identical_to_scalar']
assert long['wide_3_4_bit_identical'] and long['argmax_identical']
assert all(not v['bit_identical'] and v['metrics']['first_difference']['row']==751 and v['metrics']['different_rows']==273 for v in long['comparisons'].values())
assert long['rows']==1024 and long['effective_extents']==[256,512,768]
assert long['comparisons']['3']['counts']['raw_crops']>0 and long['ring_wraps']
assert [(e['pos'],e['sparse']) for e in long['ratio2_dispatch_changes']]==[(6617,False),(6997,True)]
summary=read(R/'free-summary.json')
configs=['integration-off','candidate-off','candidate-n1','candidate-n3']
assert summary['completed_configs']==configs and summary['comparisons']==56
assert summary['overall_server_compatibility_passed'] and all(summary['checks'].values())
assert not summary['request_errors'] and len(summary['cache_reuse'])==12
assert len(summary['probability_output_controls'])==16 and len(summary['fresh_vs_cache'])==12
reference=read(R.parent/'stage21/free-runs/snapshot-off/manifest.json')
assert reference['binary_hashes']==libs
native_index=read(R.parent/'stage21/raw-file-sha256.json')
native=R.parent/'stage21/free-runs/snapshot-off'
for name in ['dense','ratio1','ratio2']:
 tokens=read(native/(name+'-prompt.json'));assert struct.pack('<'+'i'*len(tokens),*tokens)==(R/'inputs'/(name+'.i32')).read_bytes()
tokens=read(native/'dense-fresh-response.json')['tokens'];assert struct.pack('<'+'i'*len(tokens),*tokens)==(R/'inputs/free128.i32').read_bytes()
for p in native.glob('*.json'):
 assert sha(p)==native_index[str(p.relative_to(R.parent/'stage21'))]['sha256']
for config in configs:
 d=R/'free-runs'/config;m=read(d/'manifest.json');result=read(d/'result.json')
 assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==14
 assert m['binary_hashes']==b['experiment_hashes'] and m['loaded_libraries']
 for p,h in m['loaded_libraries'].items():
  assert Path(p).parent==Path(m['library_directory']) and b['experiment_hashes'][Path(p).name]==h and sha(p)==h
 for item in result['requests']:
  request=read(d/(item['request']+'-request.json'));response=read(d/(item['request']+'-response.json'))
  assert item['prompt_sha256']==hashlib.sha256(json.dumps(request['prompt']).encode()).hexdigest()
  assert item['content_sha256']==hashlib.sha256(response['content'].encode()).hexdigest()
  assert item['tokens_sha256']==hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
  assert len(response['tokens'])==response['tokens_predicted']
 for name,count in [('dense',3033),('ratio1',3289),('ratio2',6617)]:
  full=read(d/'long-tokenize-response.json')['tokens'];prompt=read(d/(name+'-prompt.json'))
  assert len(prompt)==count and prompt==full[:count-128]+full[-128:]
for config in ['candidate-n1','candidate-n3']:
 for p in ['dense','ratio1','ratio2']:
  for state in ['fresh','cache']:
   event=summary['requests'][f'{config}/{p}-{state}']
   assert event['decode_min']=={'dense':3033,'ratio1':3289,'ratio2':6617}[p]
   assert event['decode_max']>={'dense':3072,'ratio1':3328,'ratio2':6657}[p] and event['attention']>0
   assert event['physical_plan_checked_each_decode'] and event['prefill_interventions_zero']
   acceptance=summary['acceptance'][f'{config}/{p}-{state}']
   assert acceptance['draft_n']>0 and 0<acceptance['draft_n_accepted']<=acceptance['draft_n']
source=(R/'server-context-general.cpp').read_text()
assert 'params_dft.cb_eval = nullptr;' in source and 'params_dft.cb_eval_user_data = nullptr;' in source
assert 'for (const auto & item : ds14_saved_precision) ggml_mul_mat_set_prec(item.first, item.second);' in source
assert 'ds17_down.inputs.clear();' in source and 'ds14_options.decoding = ds14_enabled && ds14_generating;' in source
for p in R.iterdir():
 if p.suffix in ['.py','.h'] or p.name=='README.md':assert p.read_bytes().isascii(),p
for p in R.glob('*.patch'):assert all(line.isascii() for line in p.read_text().splitlines() if line.startswith('+'))
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
modelstat=read(R.parent/'stage7/model-inspection.json')['models']['model'];st=Path(modelstat['path']).stat()
assert st.st_size==modelstat['size'] and st.st_mtime_ns==modelstat['mtime_ns']
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt','model-partial-summary.json','free-partial-summary.json'}
skipdirs={'__pycache__','experiment-bin','server-build'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not skipdirs&set(p.relative_to(R).parts) and p.name not in skip and p.suffix not in ['.a','.o']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':b['head'],'snapshot_hashes_verified':32,'modified_snapshot_files':[],
 'experimental_builds':1,'modified_libraries_per_build':['libllama-server-impl.so'],
 'successful_model_replays':19,'cached128_replays_exact':12,'cached1024_replays_completed':3,'cached1024_wide_numeric_comparisons_exact':0,
 'cached1024_wide_argmax_comparisons_exact':2,'cached1024_first_logit_difference_row':751,
 'cached1024_wide3_4_bit_identical':True,'overall_target_logit_compatibility_passed':False,
 'previous_full_logit_sha_controls_exact':4,'physical_raw_extents_checked':[256,512,768],
 'server_processes':4,'successful_completions':56,'http500_cache_errors':0,
 'speculative_vs_native_exact':28,'cached_speculative_vs_native_exact':6,
 'integration_off_vs_native_exact':14,'candidate_off_vs_native_exact':14,
 'n_probs5_0_pairs_exact':16,'fresh_cache_pairs_exact':12,
 'cache_requests_reusing_prompt_minus_four':12,'gpu_memory_after_mib':[15]*6,
 'model_first_shard_stat_verified':True,'production_source_changes':[],
 'failed_harness_attempts_excluded':2,'untrimmed_restore_runs_excluded':True,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
 'overall_server_compatibility_passed':True,
 'scope':'Tested single-slot prefix-cache path only. Restored1024 fixed-history logits diverge from scalar at row751 despite exact wide3/4 and argmax. Arbitrary compaction, full-session restore, multi-slot and production CUDA throughput remain untested.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
