from pathlib import Path
import json,hashlib,gzip,re,subprocess,statistics,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts';D=P/'copies8'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
r=read(D/'counts-results.json');log=(D/'counts-server.log').read_text(errors='replace');assert r['requests_complete'] and len(r['requests'])==5 and r['exit_code']==0 and not r['floor_breach'] and not r['warnings'] and not r.get('forced_kill')
assert re.search(r'asynchronous pipeline inputs = enabled \(copies = 8\)',log)
assert r['private_base_sha256']==sha(D/'candidate/libggml-base.so.0')==read(D/'build.json')['private_base_sha256'];assert any(str(D/'candidate')in x for x in r['loaded_base_maps'])
for f,h in r['config']['source_library_hashes'].items():assert sha(Path(f))==h,f
for f,h in r['config']['source_files'].items():assert sha(R/f)==h,f
for x in r['config']['model_stats']:
 s=Path(x['path']).stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==tuple(x[k]for k in ['device','inode','bytes','mtime_ns'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
ref={};cohorts={};all_requests=0
for name in ['', 'api-attribution','launch-attribution','launch-queue4x','copies8']:
 q=P/name;j=read(q/'counts-results.json');assert j['requests_complete']and j['exit_code']==0 and not j['floor_breach']and not j['warnings'];all_requests+=len(j['requests'])
 for x in j['requests']:
  t=x['response']['timings'];v=(x['tokens_sha256'],t.get('draft_n'),t.get('draft_n_accepted'));assert t['cache_n']==0 and t['prompt_n']==x['prefix'] and len(x['response']['tokens'])==16
  key=x['prefix'];ref.setdefault(key,v);assert ref[key]==v,(name,x['case'])
 controls=[x for x in j['requests']if x['kind']in['pre','post']and x['prefix']==8192];cohorts[name or 'host-scopes']={'pp8k_mean':statistics.mean(x['pp']for x in controls),'tg16_interval_mean':statistics.mean(x['tg']for x in controls),'pp_controls':[x['pp']for x in controls]}
a,b=read(D/'metadata4.json'),read(D/'metadata8.json')
for k in ['routes','routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles','context','gpu_layers','split','overrides']:assert a[k]==b[k],k
assert b['gpu_complete_bundles']==15360 and b['routed_host_bytes']==0
unit=read(D/'scheduler-results.json');assert unit['complete']and len(unit['cases'])==42
# Preserve the original incomplete controller manifest. Recovery verifies its data separately.
checks={'complete':True,'original_controller_complete':r['complete'],'original_manifest_sha256':sha(D/'counts-results.json'),'recovery_reason':'Final controller assertion searched copies=8; actual log is copies = 8. All five requests and server cleanup finished before this assertion. Original manifest and controller remain unchanged.','activation_log_regex':'asynchronous pipeline inputs = enabled \\(copies = 8\\)','private_library_loaded':True,'private_base_sha256':r['private_base_sha256'],'scheduler_cases_passed':42,'all_five_outputs_and_draft_counters_exact':True,'all_cross_cohort_outputs_and_counters_exact':True,'metadata_placement_exact':True,'gpu_expert_bundles':15360,'routed_host_bytes':0,'runtime_source_and_production_libraries_unchanged':True,'target_shard_metadata_unchanged':10,'exit_code':0,'floor_breach':[],'warnings':[],'min_free_mib':{ph:{k:round(v['min_free']/2**20,3)for k,v in r['memory'][ph].items()if isinstance(v,dict)}for ph in ['load','inference']},'cohorts':cohorts,'gain_pp8k_vs_copies4_queue4x_percent':100*(cohorts['copies8']['pp8k_mean']/cohorts['launch-queue4x']['pp8k_mean']-1),'request_checks_across_all_phases':all_requests,'quality_scope':'16-token greedy arrays and counters only; raw logits/state/rollback/NLL full-model gates are not claimed for copies8. No gain: candidate not adopted, broader gates not run.','native_scope':'Same-process pass-through pre/post controls with probe loaded but unarmed, not fresh interposer-free A/B. Queue4x fixed in the copies4/copies8 comparison. Short TG is diagnostic only.'}
save(D/'checks.json',checks)
# API analyzer reads the preserved manifest; its complete=false remains visible.
subprocess.run(['python3',str(D/'analyze.py')],cwd=R,check=True)
host=(P/'analyze.py').read_text().replace("P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts'","P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts/copies8'").replace("P/'summary.json'","P/'host-summary.json'")
(D/'analyze-host.py').write_text(host);subprocess.run(['python3',str(D/'analyze-host.py')],cwd=R,check=True)
raw=(D/'scopes.jsonl').read_bytes();packed=gzip.compress(raw,mtime=0);(D/'scopes.jsonl.gz').write_bytes(packed);assert gzip.decompress(packed)==raw;save(D/'payload.json',{'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'gzip_bytes':len(packed),'gzip_sha256':hashlib.sha256(packed).hexdigest(),'recovery_checks':'checks.json','raw_controller_complete':False})
s=read(P/'STATE.json');s.update(status='Residual diagnostic completed. Queue4x and private copies8 have no useful native PP gain; no source/default/preset/deployment/commit changes. Copies8 post-test formatting assertion recovered in separate checks.json; original incomplete manifest preserved.',next=['Read REPORT.md and copies8/checks.json; no runtime adoption.','Next prefill candidate requires exact producer/view-root/buffer-owner dependency audit before a reviewed new scheduler design.','Separate native verification sizes1..6 remains open.'],cohorts=cohorts);save(P/'STATE.json',s)
for f in ['prepare-residual-copies8.py','check-residual-copies8.py','run-residual-copies8.py','finalize-residual-screen.py']:
 shutil.copy2(Path('/tmp')/f,P/f)
print('CHECKS',json.dumps({'requests':all_requests,'gain':checks['gain_pp8k_vs_copies4_queue4x_percent'],'minFreeMiB':min(v for ph in checks['min_free_mib'].values()for v in ph.values()),'cohorts':cohorts}))
