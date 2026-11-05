from pathlib import Path
import ast
import hashlib
import json
import subprocess

R=Path('/home/sergei/Github/llama.cpp')
P=Path(__file__).resolve().parent
M=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts'
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
w=json.loads((P/'width-results.json').read_text())
n=json.loads((P/'native-results.json').read_text())
s=json.loads((P/'native-summary.json').read_text())
a=json.loads((P/'admission.json').read_text())
assert w['complete'] and w['all_outputs_state_exact']
assert n['complete'] and n['all_outputs_counters_exact'] and s['complete']
assert a['complete'] and a['admit'] and a['placement_off_on_exact']
assert s['raw_manifest_sha256']==sha(P/'native-results.json')
assert w['input_sha256']==sha(P/'inputs.json')
assert (P/'inputs.json').read_bytes()==(R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts/inputs.json').read_bytes()
assert sha(M/'candidate/libllama.so.0')==n['candidate_sha256']==w['build']['private_library_sha256']
for f,h in n['config']['source_files'].items():assert sha(R/f)==h,f
for f,h in n['config']['source_library_hashes'].items():assert sha(f)==h,f
for row in w['arms']+n['arms']:
    assert row['complete'] and row['exit_code']==0
    assert not row['floor_breach'] and not row['warnings'] and not row.get('forced_kill')
for p in P.glob('*.py'):ast.parse(p.read_text())
assert subprocess.check_output(['git','diff','--','src','ggml','common','include','tools','tests'],cwd=R,text=True)==''
gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu_processes
out=dict(complete=True,source_and_production_modules_unchanged=True,private_module_sha256=n['candidate_sha256'],
         inputs_sha256=w['input_sha256'],five_timed_processes_clean=True,gpu_compute_processes_after_run=[],
         artifact_hashes={f:sha(P/f) for f in ['width-results.json','width-summary.json','native-results.json','native-summary.json','admission.json','runtime-build.json','width-build.json']},
         no_new_runtime_source_default_changes=True,
         limits='Native complete requires exact observed outputs/counters; raw hashes/state gates and historical failure caveats are separate. No universal determinism/default promotion.')
(P/'final-checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
