#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp');SNAP=ROOT.parent/'stage8/candidate-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((ROOT/'build-manifest.json').read_text());assert build['exit_code']==0
original=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/n)==h for n,h in original.items())
assert all(sha(ROOT/'experiment-bin'/n)==h for n,h in build['experiment_hashes'].items())
assert [n for n in original if original[n]!=build['experiment_hashes'][n]]==['libllama-server-impl.so']
assert all(sha(ROOT/p)==h for p,h in build['sources'].items())
assert all(sha(p)==h for p,h in build['reused_build_inputs'].items())
replay=json.loads((ROOT/'replay-manifest.json').read_text());assert replay['exit_code']==0
assert sha(ROOT/'diagnostic-callback.h')==replay['callback_header_sha256']
assert sha(ROOT/'replay-control.cpp')==replay['source_sha256'] and sha(ROOT/'replay-control')==replay['binary_sha256']
sha_controls=json.loads((ROOT/'replay-control-summary.json').read_text());assert len(sha_controls)==3 and all(v['bit_identical'] for v in sha_controls.values())
extended=json.loads((ROOT/'extended-summary.json').read_text())
assert len(extended['controls'])==3 and all(v['bit_identical'] for v in extended['controls'].values())
assert not extended['variants']['1']['argmax_difference_indices']
m=json.loads((ROOT/'extended-replay-manifest.json').read_text());assert m['exit_code']==0 and sha(ROOT/'extended-replay.cpp')==m['source_sha256'] and sha(ROOT/'extended-replay')==m['binary_sha256'] and sha(ROOT/'diagnostic-callback.h')==m['callback_header_sha256']
summary=json.loads((ROOT/'free-summary.json').read_text());configs=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3','snapshot-n3']
loaded={};requests=0
for config in configs:
 d=ROOT/'free-runs'/config;r=json.loads((d/'result.json').read_text());m=json.loads((d/'manifest.json').read_text())
 assert r['status']=='passed' and r['server_exit_code']==0 and len(r['requests'])==(2 if config=='snapshot-n3' else 8)
 loaded.update(m['loaded_libraries']);requests+=len(r['requests'])
 for p,h in m['loaded_libraries'].items():assert sha(p)==h==m['binary_hashes'][Path(p).name]
raw={}
for directory in ['replay-output','extended-replay-output','free-runs']:
 for p in sorted((ROOT/directory).rglob('*')):
  if p.is_file():raw[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
(ROOT/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
keys=[k for k in summary['against_native'] if k.startswith(('candidate-n1','candidate-n3'))]
out={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'original_binary_hashes_verified':len(original),'experiment_binary_hashes_verified':len(original),'modified_binary_files':['libllama-server-impl.so'],'model_replays':6,'replay_logit_sha_controls_exact':6,'server_processes':len(configs),'all_server_exit_codes_zero':True,'http_completions':requests,'template_requests':18,'native_prior_token_controls_exact':sum(v['tokens_identical'] for v in summary['native_prior_controls'].values()),'integration_native_token_controls_exact':sum(v['tokens_identical'] for k,v in summary['against_native'].items() if k.startswith('integration-off-')),'candidate_off_native_token_controls_exact':sum(v['tokens_identical'] for k,v in summary['against_native'].items() if k.startswith('candidate-off-')),'candidate_spec_native_token_matches':sum(summary['against_native'][k]['tokens_identical'] for k in keys),'candidate_spec_native_token_comparisons':len(keys),'n_probs5_vs0_matching_outputs':sum(v['tokens_identical'] for v in summary['probability_output_controls'].values()),'loaded_libraries':loaded,'raw_files_hashed':len(raw),'raw_bytes_hashed':sum(v['bytes'] for v in raw.values()),'raw_index_sha256':sha(ROOT/'raw-file-sha256.json'),'production_source_status':subprocess.check_output(['git','-C',str(REPO),'diff','--name-only','HEAD','--','src','ggml','common','tools','tests'],text=True).splitlines(),'extended_first_difference_indices':{k:v['first_difference'] for k,v in extended['variants'].items()},'scope':'Isolated experimental server only. Callback execution duplicates original work; timing is not a production speedup measurement.'}
assert not out['production_source_status']
(ROOT/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
