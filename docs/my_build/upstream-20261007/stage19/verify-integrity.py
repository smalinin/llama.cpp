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
assert len(libs)==32 and all(sha(SNAP/n)==h for n,h in libs.items())
for name in ['build-manifest.json','schedule-build-manifest.json']:
 m=get(name);assert m['exit_code']==0
 assert all(sha(R/n)==h for n,h in m['sources'].items())
for name in ['run-manifest.json','schedule-run-manifest.json']:
 m=get(name);assert m['exit_code']==0 and m['snapshot_sha256']==libs
 assert sha(m['command'][0])==m['binary_sha256']
 assert sha(m['command'][2])==m['prompt_sha256']
 assert sha(m['command'][3])==m['prefix_sha256']
 if 'long_prefix_sha256' in m:assert sha(R/'long-history.i32')==m['long_prefix_sha256']
assert (R/'legacy-callback.h').read_text()==(R.parent/'stage18/v2-callback.h').read_text().replace('namespace ds14 {','namespace ds19legacy {')
for name in ['history-manifest.json','long-history-manifest.json']:
 m=get(name);p=R/('long-history.i32' if name.startswith('long-') else 'history.i32')
 assert sha(p)==m['sha256']
 for piece in m['pieces']:assert sha(piece.get('path',piece.get('source')))==piece['sha256']
loaded={}
for directory in ['boundary-output','schedule-output']:
 for line in (R/directory/'loaded-libraries.txt').read_text().splitlines():
  path=Path(line.split()[-1]);assert path.parent==SNAP and sha(path)==libs[path.name]
  loaded[str(path)]=libs[path.name]
m=get('attention-run-manifest.json')
assert sha(R.parent/'stage18/attention-replay')==m['binary_sha256']
assert sha(R.parent/'stage18/attention-replay.cpp')==m['source_sha256']
assert all(r['exit_code']==0 for r in m['runs'])
for label,case in get('attention-input-summary.json')['cases'].items():
 assert all(sha(R/'attention-inputs'/label/n)==h for n,h in case['sha256'].items())
for arch in ['ada','ampere']:
 rows=[json.loads(l) for l in (R/f'attention-{arch}/results.jsonl').read_text().splitlines()]
 assert len(rows)==10 and all(r['repeat_bit_exact'] for r in rows)
b=get('boundary-summary.json');s=get('schedule-summary.json')
assert b['controls']['trace']['bit_identical']
assert all(v['bit_identical'] for v in b['general'].values())
assert all(v['sha256']==v['expected_sha256'] for k,v in b['controls'].items() if k!='trace')
assert all(v['sha256']==v['expected_sha256'] for v in s['controls'].values())
for name in ['replay.log','schedule-replay.log']:
 lines=(R/name).read_text(errors='replace').splitlines();expected=10 if name=='replay.log' else 5
 assert sum(l.startswith('DONE ') for l in lines)==expected
 assert not any(l.startswith('FAIL ') for l in lines)
for p in R.iterdir():
 if p.suffix in ['.py','.h','.cpp'] or p.name=='README.md':assert p.read_bytes().isascii(),p
subprocess.run(['git','-C',str(REPO),'diff','--exit-code','--','.',':(exclude)docs/my_build/upstream-20261007'],check=True,capture_output=True)
model=json.loads((R.parent/'stage7/model-inspection.json').read_text())['models']['model'];stat=Path(model['path']).stat()
assert stat.st_size==model['size'] and stat.st_mtime_ns==model['mtime_ns']
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['raw-file-sha256.json','integrity.json','integrity-analysis.txt']}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'snapshot_hashes_verified':len(libs),'modified_snapshot_files':[],'model_replays':15,'boundary_replays':10,'schedule_replays':5,'frozen_attention_cases':20,'frozen_repeat_controls':20,'ada_model_capture_controls':4,'full1024_capture_sha_controls':1,'native256_sha_controls':1,'native95_sha_controls':3,'fixed_width_native1024_matches':sum(v['bit_identical'] for v in b['general'].values()),'schedule_native1024_matches':{k:v['bit_identical'] for k,v in s['histories'].items()},'long_prompt512_match':s['long_prompt']['w4']['bit_identical'],'loaded_libraries':loaded,'model_first_shard_stat_verified':True,'raw_files_hashed':len(raw),'raw_index_sha256':sha(R/'raw-file-sha256.json'),'production_source_changes':[],'scope':'Diagnostic dense attention with fresh accepted history, tested padding boundaries, bounded rollback and SWA; sparse/compaction policy and CUDA throughput remain open.'}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='loaded_libraries'},indent=2))
