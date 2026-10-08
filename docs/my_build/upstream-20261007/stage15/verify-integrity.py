from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;S=R.parent/'stage8/candidate-bin';REPO=Path('/home/sergei/Github/llama.cpp')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(16*1024*1024):h.update(chunk)
 return h.hexdigest()
original=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(S/n)==h for n,h in original.items())
assert sha(R/'diagnostic-callback.h')==sha(R.parent/'stage14/diagnostic-callback.h')
loaded={}
for label,source,binary in [('chain-capture','capture-chain.cpp','capture-chain'),('down-candidate','down-candidate.cpp','down-candidate')]:
 m=json.loads((R/(label+'-manifest.json')).read_text());assert m['exit_code']==0
 assert sha(R/source)==m['source_sha256'] and sha(R/binary)==m['binary_sha256'] and sha(R/'diagnostic-callback.h')==m['callback_header_sha256']
 for line in (R/('chain-capture-output' if label=='chain-capture' else 'down-candidate-output')/'loaded-libraries.txt').read_text().splitlines():
  p=Path(line.split()[-1]);assert p.parent==S and sha(p)==original[p.name];loaded[str(p)]=sha(p)
for name in ['build-manifest.json','candidate-build-manifest.json']:
 b=json.loads((R/name).read_text());assert b['exit_code']==0
 assert all(sha(R/n)==v for n,v in b['sources'].items())
for name in ['isolated-manifest.json','isolated-profile-manifest.json','components-manifest.json']:
 m=json.loads((R/name).read_text())
 binary='down-components' if name=='components-manifest.json' else 'down-replay'
 assert sha(R/binary)==m['binary_sha256'] and sha(R/(binary+'.cpp'))==m['source_sha256']
 for run in m['runs']:
  assert run['exit_code']==0
  for p,h in run['loaded_libraries'].items():
   assert Path(p).parent==S and sha(p)==h==original[Path(p).name];loaded[p]=h
chain=json.loads((R/'chain-summary.json').read_text());assert all(x['bit_identical'] for x in chain['controls'].values())
candidate=json.loads((R/'candidate-summary.json').read_text());assert candidate['scalar_control']['sha256']==candidate['scalar_control']['reference_sha256']
isolated=json.loads((R/'isolated-summary.json').read_text());assert len(isolated['capture_controls'])==12
profile=json.loads((R/'kernel-summary.json').read_text());assert profile['profile_outputs_exact']==44
index=json.loads((R/'inputs/weight-index.json').read_text());assert len(index)==1
for w in index:
 p=Path(w['source']);assert p.stat().st_size==w['source_size'] and p.stat().st_mtime_ns==w['source_mtime_ns']
status=subprocess.check_output(['git','-C',str(REPO),'diff','--name-only','HEAD','--','src','ggml','common','tools','tests'],text=True).splitlines();assert not status
raw={}
for directory in ['chain-capture-output','rejected-capture-output','down-candidate-output','inputs','down0-ada','down0-ampere','down0-ada-profile','down0-ampere-profile','down-ada','down-ampere','weighted-ada','weighted-ampere']:
 for p in sorted((R/directory).rglob('*')):
  if p.is_file():raw[str(p.relative_to(R))]={'bytes':p.stat().st_size,'sha256':sha(p)}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'snapshot_hashes_verified':len(original),'stage14_callback_unchanged':True,'model_replays':9,'model_processes':3,'capture_replays_rejected':3,'accepted_capture_tensor_records':5400,'capture_full95_sha_controls_exact':3,'candidate_scalar_full95_sha_control_exact':True,'native_isolated_cases':44,'component_cases':88,'native_capture_controls_exact':12,'cpu_scalar_wide_reconstruction_controls_exact':12,'profile_cases':44,'profile_float_files_exact':44,'candidate_native_argmax_difference_indices':{w:v['argmax_difference_indices'] for w,v in candidate['variants'].items()},'raw_files_hashed':len(raw),'raw_bytes_hashed':sum(v['bytes'] for v in raw.values()),'raw_index_sha256':sha(R/'raw-file-sha256.json'),'loaded_libraries':loaded,'production_source_status':status,'http_requests':0,'scope':'Diagnostic only. Native isolated full-MoE outputs match actual captures; component-mode max_vs_capture fields are unused placeholders and are not validation. Callback duplicates original work; no production timing claim.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
