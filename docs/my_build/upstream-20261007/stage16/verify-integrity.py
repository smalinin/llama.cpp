import hashlib,json,subprocess
from pathlib import Path

R=Path(__file__).resolve().parent
SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
hashes=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/n)==h for n,h in hashes.items())
builds=[]
for name in ['build-manifest.json','attention-build-manifest.json','down-cost-build-manifest.json','candidate-build-manifest.json','isolated-build-manifest.json']:
    m=json.loads((R/name).read_text())
    for f,h in m.get('sources',{}).items():assert sha(R/f)==h,(name,f)
    for b in m.get('builds',[m]):
        assert b['exit_code']==0
        source=next(Path(a) for a in b['command'] if a.endswith('.cpp'))
        if 'source_sha256' in b:assert sha(source)==b['source_sha256'],source
        binary=Path(b['command'][b['command'].index('-o')+1])
        builds.append({'manifest':name,'source':str(source),'source_sha256':sha(source),'binary_sha256':sha(binary)})
    if 'fragment_sha256' in m:assert sha(R/'attention-fragment.cpp')==m['fragment_sha256']
jobs=[]
for name in ['chain-capture-manifest.json','attention-capture-manifest.json','compressor-candidate-manifest.json']:
    m=json.loads((R/name).read_text());assert m['exit_code']==0
    cmd=m['command'];binary=Path(cmd[0]);output=Path(cmd[-1])
    assert sha(binary)==m['binary_sha256'] and sha(binary.with_suffix('.cpp'))==m['source_sha256']
    header='candidate-callback.h' if name.startswith('compressor-') else 'diagnostic-callback.h'
    assert sha(R/header)==m['callback_header_sha256']
    assert sha(cmd[2])==m['prompt_sha256'] and sha(cmd[3])==m['forced_prefix_sha256']
    loaded={l.split()[-1] for l in (output/'loaded-libraries.txt').read_text().splitlines()}
    assert loaded and all(Path(p).parent==SNAP and sha(p)==hashes[Path(p).name] for p in loaded)
    captures=list(output.rglob('tensors.jsonl'))
    jobs.append({'manifest':name,'exit_code':0,'replay_variants':len(m['variants'])*len(m['widths']),
                 'loaded_libraries':{p:sha(p) for p in sorted(loaded)},'capture_batches':len(captures),
                 'capture_records':sum(len(p.read_text().splitlines()) for p in captures)})
isolated=[]
for name in ['isolated-manifest.json','compressor-isolated-manifest.json']:
    m=json.loads((R/name).read_text())
    for run in m['runs']:
        assert run['exit_code']==0
        binary=Path(run['command'][0])
        if 'binary_sha256' in run:assert sha(binary)==run['binary_sha256'] and sha(binary.with_suffix('.cpp'))==run['source_sha256']
        elif 'binary_sha256' in m:assert sha(binary)==m['binary_sha256'] and sha(binary.with_suffix('.cpp'))==m['source_sha256']
        for p,h in run['loaded_libraries'].items():assert sha(p)==h==hashes[Path(p).name] and Path(p).parent==SNAP
        output=Path(run['command'][-1]);rows=[json.loads(l) for l in (output/'results.jsonl').read_text().splitlines()]
        assert all(r['repeat_stable'] for r in rows)
        isolated.append({'manifest':name,'output':str(output),'cases':len(rows),'repeat_stable':True})
MODE='decode-scalar-fa-upgate-hc-router-down'
for w in [2,4]:
    counts=json.loads((R/'compressor-candidate-output'/f'{MODE}-compressor-w{w}-matmul-counts.json').read_text())
    selected={k:v for k,v in counts.items() if k.startswith(('comp_state_kv-','comp_state_score-'))}
    assert set(selected)=={'comp_state_kv-2','comp_state_kv-8','comp_state_kv-14','comp_state_score-2','comp_state_score-8','comp_state_score-14','comp_state_score-20'}
    assert all(v==94 for v in selected.values())
    routed={k:v for k,v in counts.items() if k.startswith('scalar-down-')}
    assert len(routed)==40 and all(v==94 for v in routed.values())
index={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['integrity.json','raw-file-sha256.json']}
(R/'raw-file-sha256.json').write_text(json.dumps(index,indent=2)+'\n')
result={'snapshot_hashes_verified':len(hashes),'builds':builds,'model_jobs':jobs,'isolated_jobs':isolated,
        'model_replays':sum(j['replay_variants'] for j in jobs),'capture_records':sum(j['capture_records'] for j in jobs),
        'isolated_cases':sum(j['cases'] for j in isolated),'raw_files':len(index),'raw_index_sha256':sha(R/'raw-file-sha256.json'),
        'head':subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip(),
        'production_source_change':False,'installed_server_rebuilt':False}
(R/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['builds','model_jobs','isolated_jobs']},indent=2))
