#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
libs=read(R.parent/'stage8/candidate-binary-sha256.json');assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
assert all(sha(REPO/n)==h for n,h in read(R/'source-selection.json')['files'].items())
assert sha(R/'raw-layout.h')==sha(R.parent/'stage23/raw-layout.h')
for name in ['replay','projection']:
 b=read(R/f'{name}-build-manifest.json');assert b['exit_code']==0 and all(sha(R/n)==h for n,h in b['sources'].items())
 m=read(R/f'{name}-run-manifest.json');assert m['exit_code']==0 and sha(m['command'][0])==m['binary_sha256'] and m['snapshot_sha256']==libs
 loaded=(R/('model-output' if name=='replay' else 'projection-output')/'loaded-libraries.txt').read_text().splitlines();assert loaded
 for line in loaded:
  p=Path(line.split()[-1]);assert p.parent==SNAP and sha(p)==libs[p.name]
m=read(R/'replay-run-manifest.json');assert sha(m['command'][2])==m['cases_sha256'] and all(sha(p)==h for p,h in m['input_sha256'].items())
m=read(R/'projection-run-manifest.json');assert sha(R.parent/'stage23/projection-inputs/manifest.json')==m['input_manifest_sha256'] and sha(R/'projection-check.cpp')==m['source_sha256']
projection=read(R/'projection-summary.json');assert projection['variants']==projection['repeats_stable']==11 and projection['repeat_columns_exact'] and projection['scalar_sha_preserved'] and projection['wide_actual_target_column_exact']
assert projection['header_sha256']==sha(R/'general-callback.h')
model=read(R/'model-summary.json');assert model['completed_count']==12 and len(model['controls'])==6 and len(model['histories'])==6
assert model['all_controls_exact'] and model['all_history_logits_exact']
assert all(v['exact'] for v in model['indexer_selection'].values()) and all(v['exact'] for v in model['physical_traces'].values())
for label,c in model['counts'].items():
 assert (R/'model-output'/label/'logits.f32').stat().st_size==c['rows']*c['vocab']*4
 assert c['vocab']==129280
 if 'history' in label and not label.startswith('history-control'):assert c['rows']==1024 and c['prompt_tokens']==6617
assert model['controls']['cache-history-w1']['sha256']=='6b0d5aef9cbd7456149d97fa5b31e6ac4f1d56e3d41823758bb77ad444770e30'
assert model['controls']['fresh-history-w1']['sha256']=='2f04b5807f7dbe708a51c790c570d34a7b07d495946b88490c49352769ecde57'
for p in R.iterdir():
 if p.suffix in ['.py','.cpp','.h'] or p.name=='README.md':assert p.read_bytes().isascii(),p
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
modelstat=read(R.parent/'stage7/model-inspection.json')['models']['model'];st=Path(modelstat['path']).stat();assert st.st_size==modelstat['size'] and st.st_mtime_ns==modelstat['mtime_ns']
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt','model-partial-summary.json','model-partial-analysis.txt'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in skip}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':read(R/'source-selection.json')['head'],'snapshot_hashes_verified':32,'successful_model_replays':12,'previous_full_logit_sha_controls_exact':6,
 'long_cached_widths':[1,2,3,4],'long_fresh_widths':[1,2,3,4],'long_wide_scalar_comparisons_exact':6,'all_tested_full_logits_exact':True,
 'projection_frozen_variants':11,'projection_repeats_exact':11,'indexer_source_layers_checked':[2,8,14,20,24,28,32,36],
 'raw_layout_unchanged':True,'physical_traces_exact':True,'new_server_completions':0,'production_source_changes':[],
 'installed_server_changed':False,'gpu_memory_after_mib':[15]*6,'model_first_shard_stat_verified':True,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
 'scope':'Diagnostic per-query indexer projection fix; fixed single-slot histories and previous SHA controls. Free DSpark generation after this change, arbitrary cache states and production CUDA throughput remain untested.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
