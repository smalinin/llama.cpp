#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
v1=read(R/'build-manifest.json');b=read(R/'final-build-manifest.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert head==b['head'] and head.startswith('31aa1e266') and b['exit_code']==0
assert all(sha(REPO/n)==h for n,h in b['sources'].items())
assert all(sha(p)==h for p,h in b['reused_objects'].items())
assert b['before_hashes']==read(R.parent/'stage27/build-manifest.json')['candidate_hashes']
for directory,hashes in [(R.parent/'stage27/candidate-bin',b['before_hashes']),(R/'candidate-bin',v1['candidate_hashes']),(R/'final-bin',b['candidate_hashes'])]:
 assert len(hashes)==32 and all(sha(directory/n)==h for n,h in hashes.items())
assert {n for n in b['before_hashes'] if b['before_hashes'][n]!=b['candidate_hashes'][n]}=={'libllama.so','libllama.so.0','libllama.so.0.4.0'}
assert sha(REPO/'build-glm53/bin/libllama.so.0.4.0')==read(R.parent/'stage8/candidate-binary-sha256.json')['libllama.so.0.4.0']
assert subprocess.check_output(['git','diff','--',*b['sources']],cwd=REPO)==(R/'source-final.patch').read_bytes()
changed=subprocess.check_output(['git','diff','--name-only','--','.',':(exclude)docs/my_build/upstream-20261007'],cwd=REPO,text=True).splitlines();assert set(changed)==set(b['sources'])
p=read(R/'pool-run-manifest.json');assert sha(p['model'])==p['model_sha256'] and sha(R/'pool-replay.cpp')==p['source_sha256']
for name,c in p['cases'].items():
 assert sha(Path(c['library_directory'])/'libllama.so.0.4.0')==c['libllama_sha256']
 assert sha(R/(name+'.f32'))==c['output_sha256']
s=read(R/'summary.json');assert len(s['server'])==6 and all(c['exit_code']==0 for c in s['existing'].values())
for label,item in s['server'].items():
 d=R/'server-runs'/label;control=read(d/'control-manifest.json');m=read(d/'glm5next-spec-1/manifest.json')
 assert item==read(d/'glm5next-spec-1/result.json') and m['head']==head
 hashes=b['before_hashes'] if label=='before-unified' else b['candidate_hashes'] if label=='mtp-final-unified' else v1['candidate_hashes'];assert control['library_hashes']==hashes
 assert m['loaded_libraries'] and len(m['loaded_libraries'])==2
 expected=(R.parent/'stage27/candidate-bin') if label=='before-unified' else R/'final-bin' if label=='mtp-final-unified' else R/'candidate-bin'
 for path,h in m['loaded_libraries'].items():assert Path(path).parent==expected and sha(path)==h==hashes[Path(path).name]
 for path,stat in control['models'].items():
  st=Path(path).stat();assert st.st_size==stat['size'] and st.st_mtime_ns==stat['mtime_ns']
 assert control['source_hashes']==(b if label=='mtp-final-unified' else v1)['sources']
mtp=read(R/'mtp-run-manifest.json');assert sha(R/'mtp-replay.cpp')==mtp['source_sha256']
assert sha(REPO/'tests/test-llama-archs.cpp')==mtp['fixture_source_sha256']
for name,c in mtp['cases'].items():
 assert sha(Path(c['library_directory'])/'libllama.so.0.4.0')==c['libllama_sha256'] and sha(R/(name+'.f32'))==c['output_sha256']
for name,c in read(R/'final-pool-run-manifest.json').items():
 assert c['exit_code']==0 and c['libllama_sha256']==b['candidate_hashes']['libllama.so.0.4.0'] and sha(R/(name+'.f32'))==c['output_sha256']
gpu=(R/'gpu-after.csv').read_text().splitlines();assert len(gpu)==6 and all(l.endswith('15 MiB, 0 %') for l in gpu)
for p in R.iterdir():
 if p.suffix in ['.py','.cpp'] or p.name=='README.md':assert p.read_bytes().isascii(),p
skip={'raw-file-sha256.json','integrity.json','integrity-analysis.txt','partial-summary.json','partial-analysis.txt','repo-README.md'}
raw={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and not set(p.relative_to(R).parts)&{'candidate-bin','final-bin','__pycache__'} and p.name not in skip and p.suffix!='.o'}
(R/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
out={'head':head,'production_source_files':list(b['sources']),'prior_tail_fix_retained':True,'library_hashes_verified_per_directory':32,
 'capacity_abort_reproduced_cpu_and_server':True,'mtp_reuse_abort_reproduced_cpu_and_server':True,'cpu_completed_replays':13,'final_full_logit_comparisons_bit_exact':5,'rows_per_final_comparison':[46,46,12,12,4],'existing_cpu_tests_passed':2,
 'server_processes':6,'old_successful_serial_requests':3,'v1_mtp_successful_serial_requests':3,'completed_server_control_requests':48,'final_mtp_successful_requests':12,'installed_server_changed':False,'model_stat_verified':True,
 'server_pool_cache_off_comparison':s['comparisons']['server_pool_cache_off'],'raw_index_sha256':sha(R/'raw-file-sha256.json'),'raw_files_hashed':len(raw)}
(R/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
