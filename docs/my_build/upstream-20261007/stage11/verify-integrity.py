#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
head=subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip()
expected=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(ROOT.parent/'stage8/candidate-bin'/p)==h for p,h in expected.items())
assert all(sha(ROOT/'inputs'/p)==v['sha256'] for p,v in json.loads((ROOT/'inputs/manifest.json').read_text())['files'].items())
weights=json.loads((ROOT/'inputs/weight-index.json').read_text())
for w in weights:
    p=Path(w['source'])
    assert p.stat().st_size==w['source_size'] and p.stat().st_mtime_ns==w['source_mtime_ns']
    digest=hashlib.sha256()
    with p.open('rb') as f:
        f.seek(w['offset'])
        left=w['bytes']
        while left:
            data=f.read(min(left,64*1024*1024))
            assert data
            left-=len(data);digest.update(data)
    assert digest.hexdigest()==w['sha256']
processes=[]
for p in sorted(ROOT.glob('*-manifest.json')):
    if p.name=='cpu-delta-manifest.json' or p.name.startswith('initial-'): continue
    m=json.loads(p.read_text())
    assert m['exit_code']==0 and m['head']==head
    assert sha(m['command'][0])==m['binary_sha256']
    assert sha(m['command'][0]+'.cpp')==m['source_sha256']
    assert sha(ROOT/'inputs/weight-index.json')==m['weight_index_sha256']
    assert sha(ROOT/'inputs/manifest.json')==m['input_manifest_sha256']
    if 'ffn_input_sha256' in m:
        arch='ampere' if 'ampere' in p.name else 'ada'
        assert all(sha(ROOT/('ffn-'+arch)/f)==h for f,h in m['ffn_input_sha256'].items())
    processes.append(p.name)
assert len(processes)==10
m=json.loads((ROOT/'cpu-delta-manifest.json').read_text())
assert sha(ROOT/'cpu-delta')==m['binary_sha256'] and sha(ROOT/'cpu-delta.cpp')==m['source_sha256']
assert sha(ROOT/'inputs/down-expert217.bin')==m['sha256']
for name, count in [('ffn-ada',11),('ffn-ampere',11),('ffn-ada-profile',11),('ffn-ada-no-fusion',11),('down-ada',9),('down-ampere',9),('bench-ada',3),('bench-ampere',3)]:
    rows=[json.loads(l) for l in (ROOT/name/'results.jsonl').read_text().splitlines()]
    assert len(rows)==count and all(r['repeat_stable'] and r['column_max']==0 for r in rows)
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.f32','.i32','.q8_1','.bin'] and not any(x.startswith('initial-') for x in p.relative_to(ROOT).parts)}
(ROOT/'raw-artifact-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
result={'head':head,'candidate_files_verified':len(expected),'model_tensors_rehashed':len(weights),
        'model_tensor_bytes':sum(w['bytes'] for w in weights),'gpu_processes':processes,'all_exit_codes_zero':True,
        'capture_checks_exact':6,'all_repeated_outputs_stable':True,'profile_outputs_match_unprofiled':True,
        'cpu_reference_expert_sha256':m['sha256'],'raw_data_files_hashed':len(files),
        'scope':'Isolated layer2 FFN diagnosis only. Full-model free generation and DSpark TPS not rerun.'}
(ROOT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
