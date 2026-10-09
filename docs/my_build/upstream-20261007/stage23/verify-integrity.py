#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
libs=read(R.parent/'stage8/candidate-binary-sha256.json')
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
assert all(sha(REPO/n)==h for n,h in read(R/'source-selection.json')['files'].items())
for n in ['general-callback.h','raw-layout.h']:assert sha(R/n)==sha(R.parent/'stage22'/n)
for build in ['replay','focused']:
 b=read(R/f'{build}-build-manifest.json')
 assert b['exit_code']==0 and all(sha(R/n)==h for n,h in b['sources'].items())
for prefix,directory,count in [('replay','model-output',5),('focused','focused-output',4)]:
 m=read(R/f'{prefix}-run-manifest.json')
 assert m['exit_code']==0 and sha(m['command'][0])==m['binary_sha256']
 assert sha(m['command'][2])==m['cases_sha256'] and all(sha(p)==h for p,h in m['input_sha256'].items())
 assert m['snapshot_sha256']==libs
 loaded=(R/directory/'loaded-libraries.txt').read_text().splitlines();assert loaded
 for line in loaded:
  p=Path(line.split()[-1]);assert p.parent==SNAP and sha(p)==libs[p.name]
 assert len(list((R/directory).glob('*/counts.json')))==count
 for p in (R/directory).glob('*/counts.json'):
  c=read(p);assert c['rows']==1024 and c['prompt_tokens']==6617 and c['vocab']==129280
  assert (p.parent/'logits.f32').stat().st_size==1024*129280*4
capture=read(R/'capture-summary.json');focused=read(R/'focused-summary.json')
assert all(v['bit_identical'] for family in [capture['controls'],focused['controls'],focused['fresh_capture_controls']] for v in family.values())
for key in ['cache-w3','cache-w4']:
 c=capture['full_comparisons'][key];assert not c['bit_identical'] and c['argmax_identical'] and c['different_rows']==273
 assert c['first_difference']['output_index']==751 and c['first_difference']['query_position']==7367
fresh=capture['full_comparisons']['fresh-w4']
assert not fresh['bit_identical'] and not fresh['argmax_identical'] and fresh['different_rows']==743
assert fresh['first_difference']['output_index']==281 and fresh['first_difference']['query_position']==6897
assert all(not x for x in capture['captures']['cache-749']['first_differences'].values())
assert all(x[0]['owner']=='attn_out-24' for x in capture['captures']['cache-750']['first_differences'].values())
assert not any((R/'focused-output').glob('fresh-control-*/batch*'))
isolated=read(R/'isolated-summary.json')
assert isolated['frozen_variants']==23 and isolated['all_repeats_stable']
assert isolated['projection']['variants']==11 and isolated['projection']['capture_w1']['bit_identical'] and isolated['projection']['capture_w4']['bit_identical']
assert not isolated['projection']['frozen_width']['bit_identical']
assert all(all(v for k,v in c.items() if k!='last_selected') for c in isolated['indexer'].values())
assert all(c['capture_exact'] for c in isolated['attention'].values())
assert isolated['indexer_inputs']=={'q.f32':True,'weights.f32':False,'mask.f16':True,'visible_keys_equal':True,'visible_rows':7368}
assert isolated['topk_by_target']['749']['24']['set_equal']
assert isolated['topk_by_target']['750']['24']['only_scalar']==[3607] and isolated['topk_by_target']['750']['24']['only_wide']==[1106]
assert focused['targets']['750']['attention_inputs']['only_scalar_visible']==[4375] and focused['targets']['750']['attention_inputs']['only_wide_visible']==[1874]
assert focused['targets']['750']['attention_inputs']['q_bit_exact'] and focused['targets']['750']['attention_inputs']['common_visible_k_bit_exact']
for name in ['projection','indexer','attention']:
 m=read(R/f'{name}-run-manifest.json')
 assert m['exit_code']==0 and sha(m['command'][0])==m['binary_sha256'] and sha(Path(m['command'][0]).with_suffix('.cpp'))==m['source_sha256']
 assert m['model_device']==3 and m['snapshot_sha256']==libs
 inp=R/'projection-inputs/manifest.json' if name=='projection' else R/f'{name}-input-summary.json'
 assert sha(inp)==m['input_manifest_sha256']
 if name=='attention':
  deps=read(R/'attention-dependencies.json');assert deps['exit_code']==0 and deps['resolved_ggml_libraries']
  for path,h in deps['resolved_ggml_libraries'].items():
   q=Path(path);assert q.parent==SNAP and sha(q)==h==libs[q.name]
 else:
  loaded=(R/f'{name}-output/loaded-libraries.txt').read_text().splitlines();assert loaded
  for line in loaded:
   q=Path(line.split()[-1]);assert q.parent==SNAP and sha(q)==libs[q.name]
projection=read(R/'projection-inputs/manifest.json');weight_path=Path(projection['source']);st=weight_path.stat()
assert st.st_size==projection['model_size'] and st.st_mtime_ns==projection['model_mtime_ns']
with weight_path.open('rb') as weight:
 weight.seek(projection['offset']);assert hashlib.sha256(weight.read(projection['bytes'])).hexdigest()==projection['sha256']
assert all(sha(R/'projection-inputs'/n)==v['sha256'] for n,v in projection['files'].items())
for name in ['indexer','attention']:
 for label,c in read(R/f'{name}-input-summary.json').items():
  assert all(sha(R/f'{name}-inputs'/label/n)==h for n,h in c['files'].items())
indexer_build=read(R/'indexer-build-manifest.json');assert indexer_build['exit_code']==0 and sha(R/'indexer-replay.cpp')==indexer_build['source_sha256']
for p in R.iterdir():
 if p.suffix in ['.py','.h','.cpp'] or p.name=='README.md':assert p.read_bytes().isascii(),p
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
modelstat=read(R.parent/'stage7/model-inspection.json')['models']['model'];st=Path(modelstat['path']).stat()
assert st.st_size==modelstat['size'] and st.st_mtime_ns==modelstat['mtime_ns']
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in skip}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':read(R/'source-selection.json')['head'],'snapshot_hashes_verified':32,'production_source_changes':[],
 'successful_model_replays':9,'stage22_full_logit_sha_controls_exact':5,'fresh_capture_no_capture_controls_exact':2,
 'cache_first_difference':capture['full_comparisons']['cache-w4']['first_difference'],'fresh_first_difference':fresh['first_difference'],
 'overall_target_logit_compatibility_passed':False,'gpu_memory_after_mib':[15]*6,'model_first_shard_stat_verified':True,
 'frozen_variants':23,'frozen_repeats_bit_exact':23,'cached_cause':'BF16 indexer projection width dependence; one changed top-k row in layer24',
 'fresh_first_cause_localized':False,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
 'scope':'Single-slot fixed-token diagnostic history, F16 KV. Cached and fresh width dependence remain. No production patch, installed-server change or speedup measurement.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
