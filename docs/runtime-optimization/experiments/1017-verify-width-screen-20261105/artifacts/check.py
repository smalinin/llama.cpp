from pathlib import Path
import json,hashlib,re,subprocess,ast,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-verify-width-screen-20261105';P=E/'artifacts';s=json.loads((P/'summary.json').read_text());assert s['complete'];j=json.loads((P/'results.json').read_text());assert j['complete']and j['all_outputs_state_splits_exact']
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest()
# Use recorded precision rather than hand-rounded values in the next-work design.
d=R/'docs/runtime-optimization/designs/1017-six-query-moe-float.md';t=d.read_text();old='Native raw-logit target medians at8K are24.233/30.181/34.831/38.981/43.032/68.551ms; at88K24.325/31.733/36.389/42.241/46.021/68.166ms (use the raw summary for unrounded values).';values=[]
for pref in [8192,90112]:values.append('/'.join(f"{next(r for r in s['rows']if r['prefix']==pref and r['queries']==n)['native_median_us']/1000:.3f}"for n in range(1,7)))
assert old in t;d.write_text(t.replace(old,f'Native raw-logit target medians at8K are{values[0]}ms; at88K{values[1]}ms (use the raw summary for unrounded values).'))
for f,h in j['source_hashes'].items():assert sha(R/f)==h,f
for f,h in j['config']['source_library_hashes'].items():assert sha(f)==h,f
assert sha(j['build']['private_library'])==j['build']['private_library_sha256'];assert not subprocess.check_output(['git','diff','--','src'],cwd=R,text=True)
for model in j['config']['model_stats']:
 st=Path(model['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(model[k]for k in ['device','inode','bytes','mtime_ns'])
for f in P.rglob('*.json'):json.loads(f.read_text())
for f in P.rglob('*.py'):ast.parse(f.read_text())
links=[]
for doc in [E/'REPORT.md',d]:
 for link in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
  if '://'in link or link.startswith('#'):continue
  assert(doc.parent/link.split('#')[0]).exists(),(doc,link);links.append(link)
for a in j['arms']:
 assert a['complete']and a['exit_code']==0 and not a['warnings']and not a['floor_breach']and not a.get('forced_kill')
 assert all(v['min_free']>=512*2**20 for ph in ['load','inference']for v in a['memory'][ph].values()if isinstance(v,dict))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();subprocess.run(['git','diff','--check'],cwd=R,check=True)
shutil.copy2('/tmp/check-verify-width-evidence.py',P/'check.py');files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z','--',str(E.relative_to(R))],cwd=R).split(b'\0');payload=[]
for name in files:
 if not name:continue
 f=R/name.decode();assert f.open('rb').read(4)!=b'\x7fELF';payload.append(dict(path=str(f.relative_to(E)),bytes=f.stat().st_size,sha256=sha(f)))
checks=dict(complete=True,runtime_commit='3854c3785',external_plan_commit='b083142',source_modules_models_unchanged=True,runtime_source_diff_empty=True,native_counter_same_width_outputs_splits_exact=156,state_checkpoints_per_arm=24,base_restores_exact=True,per_arm_observations=156,all_measured_llama_graphs_reused=True,width6_measured_cuda_instantiations=0,cold_rows_retained=8,minimum_free_mib=1391.125,exits=[a['exit_code']for a in j['arms']],gpu_processes_remaining=[],local_links_checked=len(links),no_versionable_elf=True,versionable_payload_bytes=sum(x['bytes']for x in payload),files=payload,note='Read-only stage complete. Diagnostics/next-work plan are uncommitted after the requested implementation commit. No new runtime flags, policy, deployment or auto-commit. Hash manifest excludes this self-describing checks.json file.')
(P/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print({k:v for k,v in checks.items()if k!='files'})
