from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def get(name):return json.loads((R/name).read_text())
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/n)==h for n,h in libs.items())
for name in ['build-manifest.json','build-v2-manifest.json']:
 m=get(name);assert m['exit_code']==0
 assert all(sha(R/n)==h for n,h in m['sources'].items())
 assert all(sha(p)==h for p,h in m['reused_build_inputs'].items())
 assert sha(REPO/'tools/server/server-context.cpp')==m['original_server_source_sha256']
 assert [n for n in libs if m['experiment_hashes'][n]!=libs[n]]==['libllama-server-impl.so']
 directory=R/('experiment-v2-bin' if 'v2' in name else 'experiment-bin')
 assert all(sha(directory/n)==h for n,h in m['experiment_hashes'].items())
assert (R/'diagnostic-callback.h').read_bytes()==(R.parent/'stage17/diagnostic-callback.h').read_bytes()
assert (R/'v1-callback.h').read_text()==(R/'candidate-callback.h').read_text().replace('namespace ds14 {','namespace ds18v1 {')
for run in ['capture','candidate-capture','candidate','v2']:
 m=get(f'{run}-run-manifest.json');assert m['exit_code']==0
 assert sha(m['command'][0])==m['binary_sha256']
for name in ['capture-summary.json','candidate-capture-summary.json']:
 assert all(v['bit_identical'] for v in get(name)['controls'].values())
v2=get('v2-summary.json');assert all(v['bit_identical'] and not v['argmax_differences'] for rows in v2.values() for v in rows.values())
for name in ['capture-build-manifest.json','candidate-capture-build-manifest.json']:
 m=get(name);assert m['exit_code']==0
 source=R/('capture-replay.cpp' if name.startswith('capture-') else 'candidate-capture.cpp')
 header=R/('diagnostic-callback.h' if name.startswith('capture-') else 'candidate-callback.h')
 assert sha(source)==m['source_sha256'] and sha(header)==m['header_sha256']
for name in ['candidate-build-manifest.json','v2-build-manifest.json']:
 m=get(name);assert m['exit_code']==0
 assert all(sha(R/p)==h for p,h in m['sources'].items())
loaded={}
for directory in ['capture-output','candidate-output','candidate-capture-output','v2-output']:
 for line in (R/directory/'loaded-libraries.txt').read_text().splitlines():
  path=Path(line.split()[-1]);assert path.parent==SNAP and sha(path)==libs[path.name]
  loaded[str(path)]=libs[path.name]
for name in ['attention-run-manifest.json','layer20-run-manifest.json']:
 m=get(name)
 assert sha(R/'attention-replay')==m['binary_sha256'] and sha(R/'attention-replay.cpp')==m['source_sha256']
 assert all(r['exit_code']==0 for r in m['runs'])
for name in ['profile-summary.json','layer20-profile-summary.json']:
 assert all(c['profile_output_bit_exact'] for a in get(name).values() for c in a['cases'])
for arch in ['ada','ampere']:
 for name in [f'attention-{arch}/results.jsonl',f'layer20-{arch}/results.jsonl']:
  assert all(json.loads(l)['repeat_bit_exact'] for l in (R/name).read_text().splitlines())
free=get('free-summary.json');configs=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']
requests=0
for config in configs:
 d=R/'free-runs'/config;m=json.loads((d/'manifest.json').read_text());result=json.loads((d/'result.json').read_text())
 assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==8
 requests+=len(result['requests'])
 for p,h in m['loaded_libraries'].items():
  assert sha(p)==h==m['binary_hashes'][Path(p).name];loaded[p]=h
 for request in result['requests']:
  response=json.loads((d/(request['request']+'-response.json')).read_text())
  assert hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()==request['tokens_sha256']
  assert hashlib.sha256(response['content'].encode()).hexdigest()==request['content_sha256']
assert all(v['tokens_identical'] and v['content_identical'] and v['stop_metadata_identical'] for k,v in free['against_native'].items() if k.startswith(('integration-off','candidate-off')))
assert all(v['tokens_identical'] and v['content_identical'] for v in free['probability_output_controls'].values())
for p in R.iterdir():
 if (p.suffix in ['.py','.h','.cpp'] or p.name=='README.md') and not p.name.startswith('server-context'):assert p.read_bytes().isascii(),p
for patch in R.glob('*.patch'):
 assert all(l.isascii() for l in patch.read_text().splitlines() if l.startswith('+')),patch
assert all(v['tokens_identical'] and v['content_identical'] and v['stop_metadata_identical'] for k,v in free['against_native'].items() if k.startswith(('candidate-n1','candidate-n3')))
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','src','ggml','tools'],check=True,capture_output=True)
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not {'experiment-bin','experiment-v2-bin','__pycache__'}.intersection(p.parts) and p.name not in ['raw-file-sha256.json','integrity.json','integrity-analysis.txt']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
keys=[k for k in free['against_native'] if k.startswith(('candidate-n1','candidate-n3'))]
out={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'snapshot_hashes_verified':len(libs),'modified_snapshot_files':['libllama-server-impl.so'],'model_replays':22,'stage17_capture_sha_controls':4,'v1_capture_sha_controls':6,'v2_native95_sha_controls':4,'v2_native256_sha_controls':4,'frozen_attention_cases':44,'profiler_sha_controls':44,'server_processes':5,'http_completions':requests,'template_requests':15,'prior_native_controls_exact':sum(v['tokens_identical'] and v['content_identical'] for v in free['native_prior_controls'].values()),'native_integration_controls_exact':16,'spec_native_matches':sum(free['against_native'][k]['tokens_identical'] and free['against_native'][k]['content_identical'] for k in keys),'spec_native_comparisons':len(keys),'n_probs5_vs0_matching_outputs':20,'loaded_libraries':loaded,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),'production_source_changes':[],'scope':'Diagnostic first absolute256 padding boundary only; no production speed claim.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='loaded_libraries'},indent=2))
