import hashlib,json,subprocess
from pathlib import Path

R=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp');SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
build=json.loads((R/'build-manifest.json').read_text());assert build['exit_code']==0
hashes=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/n)==h for n,h in hashes.items())
assert all(sha(R/'experiment-bin'/n)==h for n,h in build['experiment_hashes'].items())
assert [n for n,h in hashes.items() if build['experiment_hashes'][n]!=h]==['libllama-server-impl.so']
assert all(sha(R/n)==h for n,h in build['sources'].items())
assert all(sha(p)==h for p,h in build['reused_build_inputs'].items())
assert sha(REPO/'tools/server/server-context.cpp')==build['original_server_source_sha256']
assert sha(R.parent/'stage16/candidate-callback.h')==build['base_callback_sha256']
assert sha(R.parent/'stage15/down-candidate.cpp')==build['down_source_sha256']
base=(R.parent/'stage15/down-candidate.cpp').read_text()
body=base[base.index('struct DownInput'):base.index('static void precision_replay(')]
assert (R/'diagnostic-callback.h').read_text()==(R.parent/'stage16/candidate-callback.h').read_text()+'\nnamespace ds14 {\n'+body+'}\n'
replay=json.loads((R/'replay-manifest.json').read_text());assert replay['exit_code']==0
assert sha(R/'diagnostic-callback.h')==replay['callback_header_sha256']
assert sha(R/'replay-control.cpp')==replay['source_sha256'] and sha(R/'replay-control')==replay['binary_sha256']
controls=json.loads((R/'replay-control-summary.json').read_text())
assert len(controls)==4 and all(v['bit_identical'] for v in controls.values())
explain=json.loads((R/'explain-summary.json').read_text())
assert len(explain['controls'])==2 and all(v['all256_native_argmax_exact'] for v in explain['controls'].values())
eb=json.loads((R/'explain-build-manifest.json').read_text());em=json.loads((R/'explain-replay-manifest.json').read_text())
assert eb['exit_code']==em['exit_code']==0
assert sha(R/'explain-replay.cpp')==eb['source_sha256']==em['source_sha256']
assert sha(R/'explain-replay')==em['binary_sha256']
assert sha(R/'diagnostic-callback.h')==eb['callback_sha256']==em['callback_header_sha256']
assert sha(R/'explain-native.i32')==eb['forced_prefix_sha256']==em['forced_prefix_sha256']
assert sha(R/'free-runs/snapshot-off/explain-prompt.txt')==eb['prompt_sha256']==em['prompt_sha256']
assert sha(R/'free-runs/snapshot-off/explain-greedy-2-response.json')==eb['native_response_sha256']
loaded={l.split()[-1] for l in (R/'replay-output/loaded-libraries.txt').read_text().splitlines()}
loaded.update(l.split()[-1] for l in (R/'explain-replay-output/loaded-libraries.txt').read_text().splitlines())
assert loaded and all(Path(p).parent==SNAP and sha(p)==hashes[Path(p).name] for p in loaded)
summary=json.loads((R/'free-summary.json').read_text())
configs=['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3']
requests=0;libraries={}
for config in configs:
    d=R/'free-runs'/config;result=json.loads((d/'result.json').read_text());m=json.loads((d/'manifest.json').read_text())
    assert result['status']=='passed' and result['server_exit_code']==0 and len(result['requests'])==8
    requests+=len(result['requests']);libraries.update(m['loaded_libraries'])
    for p,h in m['loaded_libraries'].items():assert sha(p)==h==m['binary_hashes'][Path(p).name]
    for record in result['requests']:
        response=json.loads((d/(record['request']+'-response.json')).read_text())
        assert hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()==record['tokens_sha256']
        assert hashlib.sha256(response['content'].encode()).hexdigest()==record['content_sha256']
keys=[k for k in summary['against_native'] if k.startswith(('candidate-n1','candidate-n3'))]
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and 'experiment-bin' not in p.parts and p.name not in ['raw-file-sha256.json','integrity.json']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
     'snapshot_hashes_verified':len(hashes),'experiment_hashes_verified':len(hashes),'modified_binary_files':['libllama-server-impl.so'],
     'callback_components_match_stage16':True,'model_replays':12,'native95_logit_sha_controls_exact':4,'explain_scalar256_argmax_controls_exact':2,
     'server_processes':5,'http_completions':requests,'template_requests':15,'all_server_exit_codes_zero':True,
     'prior_native_controls_exact':sum(v['tokens_identical'] and v['content_identical'] for v in summary['native_prior_controls'].values()),
     'native_integration_controls_exact':sum(v['tokens_identical'] and v['content_identical'] for k,v in summary['against_native'].items() if k.startswith(('integration-off-','candidate-off-'))),
     'spec_native_matches':sum(summary['against_native'][k]['tokens_identical'] and summary['against_native'][k]['content_identical'] for k in keys),
     'spec_native_comparisons':len(keys),'n_probs5_vs0_matching_outputs':sum(v['tokens_identical'] and v['content_identical'] for v in summary['probability_output_controls'].values()),
     'spec_native_stop_metadata_matches':sum(summary['against_native'][k]['stop_metadata_identical'] for k in keys),
     'loaded_libraries':libraries,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
     'production_source_changes':subprocess.check_output(['git','-C',str(REPO),'diff','--name-only','HEAD','--','src','ggml','common','tools','tests'],text=True).splitlines(),
     'scope':'Diagnostic server callbacks duplicate original operations and include CPU copies, allocation and synchronization. No production throughput claim.'}
assert not out['production_source_changes']
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='loaded_libraries'},indent=2))
