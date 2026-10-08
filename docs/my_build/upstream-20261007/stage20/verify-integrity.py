from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def get(name):return json.loads((R/name).read_text())
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text());assert len(libs)==32
assert all(sha(SNAP/n)==h for n,h in libs.items())
build=get('build-manifest.json');run=get('run-manifest.json')
assert build['exit_code']==run['exit_code']==0
assert all(sha(R/n)==h for n,h in build['sources'].items())
assert sha(R/'general-callback.h')==sha(R.parent/'stage19/general-callback.h')==build['reused_header_sha256']
assert sha(run['command'][0])==run['binary_sha256'] and run['snapshot_sha256']==libs
assert sha(run['command'][2])==run['prompt_sha256'] and sha(run['command'][3])==run['prefix_sha256']
assert all(sha(REPO/n)==h for n,h in get('source-selection.json')['files'].items())
loaded={}
for line in (R/'model-output/loaded-libraries.txt').read_text().splitlines():
 p=Path(line.split()[-1]);assert p.parent==SNAP and sha(p)==libs[p.name];loaded[str(p)]=libs[p.name]
history=(R.parent/'stage19/history.i32').read_bytes();base=None
for name,length in [('k4096',3033),('ratio1',3289),('ratio2',6617)]:
 d=R/'model-output'/name;prompt=(d/'prompt.i32').read_bytes()
 assert len(prompt)==length*4 and (d/'forced.i32').read_bytes()==history[:96*4]
 if base is None:base=prompt[:23*4]
 assert prompt[:23*4]==base
 assert prompt[23*4:]==(history*((length-23+1023)//1024))[:(length-23)*4]
 chunks=json.loads((d/'prefill-chunks.json').read_text());assert sum(chunks)==length and max(chunks)<=2048 and chunks[-2:]==[512,4]
model=get('model-summary.json');hp=get('model-output/model-hparams.json')
assert hp['swa']==128 and hp['indexer_top_k']==2048
assert all(v['bit_identical'] for ws in model['windows'].values() for v in ws.values())
assert model['controls']['trace-ratio2-w4']['bit_identical']
assert all(v['sha256']==v['expected_sha256'] for k,v in model['controls'].items() if k.startswith('baseline'))
for name,layer,boundary in [('ratio1',20,3328),('ratio2',2,6657)]:
 rows=model['windows'][name]['1']['transitions'][str(layer)]
 assert [r['absolute_position'] for r in rows if r['query_sparse_from_predicate']][0]==boundary
 assert all(r['n_kv_max']==2176 and r['raw_visible']==128 and r['comp_visible']==2048 for r in rows)
assert not any(r['query_sparse_from_predicate'] for r in model['windows']['k4096']['1']['transitions']['20'])
lines=(R/'replay.log').read_text(errors='replace').splitlines()
assert sum(l.startswith('DONE ') for l in lines)==13 and not any(l.startswith('FAIL ') for l in lines)
for name in ['attention-run-manifest.json','profile-run-manifest.json']:
 m=get(name);assert sha(m['binary'])==m['binary_sha256'] and sha(m['source'])==m['source_sha256']
 assert len(m['runs'])==3 and all(r['exit_code']==0 for r in m['runs'])
 if name.startswith('profile'):assert all(r['export_exit_code']==0 for r in m['runs'])
inputs=get('attention-input-summary.json')['cases'];assert len(inputs)==22
for label,c in inputs.items():assert all(sha(R/'attention-inputs'/label/n)==h for n,h in c['sha256'].items())
for arch in ['ada0','ada2','ampere']:
 for prefix in ['attention-','attention-profile-']:
  rows=[json.loads(l) for l in (R/f'{prefix}{arch}/results.jsonl').read_text().splitlines()]
  assert len(rows)==22 and all(r['repeat_bit_exact'] for r in rows)
attention=get('attention-summary.json')['architectures'];profile=get('profile-summary.json')
assert all(v['bit_identical'] for a in attention.values() for v in a['model_capture_controls'].values())
assert all(v['windows'][n][k]['bit_identical'] for v in attention.values() for n in ['k4096','ratio1','ratio2'] for k in ['crop_vs_native','pad_vs_wide','native_dense_vs_native'])
assert all(c['profile_output_bit_exact'] for a in profile.values() for c in a['cases'])
assert all(a['main_calls']==44 and a['compaction_calls']==16 and sum(c['sparse_kernel'] for c in a['cases'])==8 for a in profile.values())
for p in R.iterdir():
 if p.suffix in ['.py','.h','.cpp'] or p.name=='README.md':assert p.read_bytes().isascii(),p
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
m=json.loads((R.parent/'stage7/model-inspection.json').read_text())['models']['model'];st=Path(m['path']).stat();assert st.st_size==m['size'] and st.st_mtime_ns==m['mtime_ns']
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['raw-file-sha256.json','integrity.json','integrity-analysis.txt']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'snapshot_hashes_verified':32,'modified_snapshot_files':[],'diagnostic_header_unchanged':True,'model_replays':13,'fixed_history_windows':3,'wide_vs_scalar_comparisons_exact':7,'trace_sha_controls':1,'native95_sha_controls':2,'frozen_attention_variants':66,'frozen_repeat_controls':66,'profile_sha_controls':66,'model_capture_comparisons_exact':sum(len(a['model_capture_controls']) for a in attention.values()),'unique_model_captures':10,'profile_main_calls':132,'profile_compaction_calls':48,'profile_sparse_cases':24,'profile_dense_cases':42,'loaded_libraries':loaded,'model_first_shard_stat_verified':True,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),'gpu_memory_after_mib':[15]*6,'production_source_changes':[],'scope':'Same Stage19 diagnostic policy across tested dense/sparse windows; fixed histories only, no expanded HTTP or production speed claim.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='loaded_libraries'},indent=2))
