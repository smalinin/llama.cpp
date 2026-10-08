#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
head=subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip()
SNAP=ROOT.parent/'stage8/candidate-bin'
hashes=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/name)==h for name,h in hashes.items())
inputs=ROOT.parent/'stage11/inputs'
assert all(sha(inputs/f)==m['sha256'] for f,m in json.loads((inputs/'manifest.json').read_text())['files'].items())
weights=json.loads((inputs/'weight-index.json').read_text())
for w in weights:
    p=Path(w['source']);st=p.stat()
    assert st.st_size==w['source_size'] and st.st_mtime_ns==w['source_mtime_ns']
    h=hashlib.sha256()
    with p.open('rb') as f:
        f.seek(w['offset']);left=w['bytes']
        while left:
            data=f.read(min(left,64*1024*1024));assert data
            left-=len(data);h.update(data)
    assert h.hexdigest()==w['sha256']
processes=[]
for p in ROOT.glob('*-manifest.json'):
    m=json.loads(p.read_text())
    assert m['head']==head and m['exit_code']==0
    assert sha(m['command'][0])==m['binary_sha256']
    assert sha(m['command'][0]+'.cpp')==m['source_sha256']
    if p.name=='target-ffn-manifest.json':
        assert sha(m['command'][2])==m['prompt_sha256'] and sha(m['command'][3])==m['forced_prefix_sha256']
    else:
        assert sha(inputs/'manifest.json')==m['input_manifest_sha256']
        assert sha(inputs/'weight-index.json')==m['weight_index_sha256']
    processes.append(p.name)
assert len(processes)==4
libs=sorted({l.split()[-1] for l in (ROOT/'target-ffn-output/loaded-libraries.txt').read_text().splitlines()})
assert libs and all(Path(p).parent==SNAP and sha(p)==hashes[Path(p).name] for p in libs)
controls=json.loads((ROOT/'controls-summary.json').read_text())
target=json.loads((ROOT/'target-ffn-output/summary.json').read_text())
profile=json.loads((ROOT/'kernel-summary.json').read_text())
assert sum(controls['native_matches_stage11'].values())==154
assert len(target['controls'])==8 and all(v['bit_identical'] for v in target['controls'].values())
assert profile['profile_output_files_match']==154 and profile['profile_results_identical']
raw={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.f32','.i32','.sqlite','.nsys-rep']}
(ROOT/'raw-artifact-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
result={'head':head,'candidate_files_verified':len(hashes),'weights_rehashed':len(weights),'weight_bytes':sum(w['bytes'] for w in weights),
        'processes':sorted(processes),'all_exit_codes_zero':True,'native_ffn_data_controls_exact':154,
        'profile_data_controls_exact':154,'model_logit_sha_controls_exact':8,'model_replays':12,
        'loaded_libraries':{p:sha(p) for p in libs},'raw_data_files_hashed':len(raw),
        'scope':'Diagnostic graph change only, no production library/server change. Full-model greedy compatibility still fails; no free DSpark generation or HTTP TPS rerun.'}
(ROOT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
