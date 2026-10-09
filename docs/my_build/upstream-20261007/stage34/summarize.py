from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
labels=['qwen-final-v2','glm5next-final','glm-dsa-final','deepseek-final','native-final']
def load(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
runs=load(R/'slot-runs.json');assert len(runs)==5 and all(x['exit_code']==0 for x in runs),runs
res={'base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'mmf':{},'baseline':load(R/'qwen-before/result.json'),'slots':{},'regressions':{}}
for name in ['strides-ada','strides-rtx3090']:
 cases=[json.loads(x) for x in (R/(name+'.log')).read_text().splitlines() if x.startswith('{')]
 assert len(cases)==288 and all(x['passed'] for x in cases)
 res['mmf'][name]={'cases':len(cases),'passed':len(cases),'max_nmse':max(x['nmse'] for x in cases),'max_abs':max(x['max_abs'] for x in cases),'log_sha256':sha(R/(name+'.log'))}
for name in labels:
 data=load(R/name/'result.json');assert data['tokens_equal'] and data['bytes_equal'] and data['expected_cached']==data['restored_cache']==data['control_cache']
 if 'restart_equal' in data:assert data['restart_equal'] and data['restart_cache']==data['expected_cached']
 data['generation_requests']=len(list((R/name).glob('*-response.json')))-sum(1 for p in (R/name).glob('*-response.json') if 'timings' not in load(p) or 'predicted_n' not in load(p)['timings'])
 res['slots'][name]=data
assert '4/4 tests passed' in (R/'backend-regression.log').read_text()
assert 'GGML_ASSERT(stride_row % 2 == 0) failed' in (R/'backend-regression-before.log').read_text()
assert '1 passed' in (R/'pytest-final.log').read_text()
res['regressions']={'mmf_before_assertion':True,'mmf_regression_cases_passed':4,'pytest_passed':1,'native_reads_speculative_file':load(R/'native-compat/result.json')}
build=load(R/'build-manifest.json')
for p,h in build['sources'].items():assert sha(REPO/p)==h,p
res['sources']={p:sha(REPO/p) for p in subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines() if p.startswith(('ggml/','tools/','tests/'))}
res['build_library_hashes']=build['hashes'];(R/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k not in ['build_library_hashes','sources']},indent=2))
