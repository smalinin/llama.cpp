from pathlib import Path
import json,hashlib,subprocess,re,math
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105';P=E/'artifacts'
def sha(f):
 with Path(f).open('rb')as h:return hashlib.file_digest(h,'sha256').hexdigest()
b=json.loads((P/'final-build.json').read_text());n=json.loads((P/'native-results.json').read_text());u=json.loads((P/'unit-manifest.json').read_text());q=json.loads((P/'precision-results.json').read_text());out={}
assert b['source_sha256']==n['source_sha256']==u['source_sha256']==sha(R/'ggml/src/ggml-backend.cpp');assert b['candidate_sha256']==n['candidate_sha256']==u['candidate_sha256']==q['candidate_sha256']==sha(P/'candidate/libggml-base.so.0')
assert sha(R/'build-aliceai/bin/libggml-base.so.0')==b['production_base_sha256'];assert sha(P/'native-initial-ggml-backend.cpp')==b['source_sha256'];subprocess.run(['git','apply','--reverse','--check',str(P/'native-initial-runtime.patch')],cwd=R,check=True)
out['source_measured_native_unit_precision_identical']=True;out['production_library_unchanged']=True
for f,h in n['config']['source_library_hashes'].items():assert sha(f)==h,f
for f,h in n['config']['source_files'].items():assert sha(R/f)==h,f
for r in n['config']['model_stats']:
 st=Path(r['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(r['device'],r['inode'],r['bytes'],r['mtime_ns']),r['path']
d=n['config']['draft'];assert Path(d['path']).resolve()==Path(d['resolved'])and Path(d['path']).stat().st_size==d['bytes'];assert sha(d['path'])==d['sha256']
out['target_shard_metadata_unchanged']=len(n['config']['model_stats']);out['draft_sha256_unchanged']=True;out['other_runtime_source_library_hashes_unchanged']=True
assert n['complete']and n['all_outputs_counters_exact']and sum(len(a['requests'])for a in n['arms'])==24
out['native_requests_exact']=24;out['native_exit_codes']=[a['exit_code']for a in n['arms']];assert out['native_exit_codes']==[0,0,0]
res=[];out['precision']=[]
assert q['complete']and q['all_groups_exact']and len(q['arms'])==2
for a in q['arms']:
 p=P/('precision-'+a['flag']);r=json.loads((p/'results.json').read_text());assert r['complete']and len(r['cases'])==12 and len(r['faults'])==2 and all(f['recovered']for f in r['faults']);assert a['exit_code']==0
 rows={(v['corpus'],v['key']):v['result']for v in r['cases']};res.append(rows)
 for k,v in rows.items():
  if k[1]=='rollback':assert all(x['target_replay']and x['draft_replay']for x in v)
  if k[1]=='nll128':assert len(v['rows'])==128 and math.isfinite(v['mean_nll'])
 inside=False;spans=0;unexpected=[]
 for line in(p/'server.log').read_text(errors='replace').splitlines():
  if line=='FAULT_BEGIN':assert not inside;inside=True;spans+=1
  elif line=='FAULT_END':assert inside;inside=False
  elif ' E 'in line and not inside:unexpected.append(line)
 assert not inside and spans==2 and not unexpected
 out['precision'].append({'flag':a['flag'],'groups':len(rows),'exit':a['exit_code'],'fault_retry_spans':spans,'unexpected_errors':unexpected})
assert res[0]==res[1];out['precision_paired_groups_exact']=12
out['nll']={k[0]:{'off':v['mean_nll'],'on':res[1][k]['mean_nll'],'delta':0,'positions':128}for k,v in res[0].items()if k[1]=='nll128'}
out['memory_min_free_mib']={}
for group,arms in [('native',n['arms']),('precision',q['arms'])]:
 for a in arms:
  assert not a['floor_breach']and not a['warnings']and not a.get('forced_kill')
  v=min(d['min_free']for ph in ['load','inference']for d in a['memory'][ph].values()if isinstance(d,dict))/2**20;assert v>=512;out['memory_min_free_mib'][group+'-'+a.get('label',a['flag'])]=v
unit=json.loads((P/'scheduler-results.json').read_text());assert unit['complete']and len(unit['cases'])==42;out['scheduler_cases']=42
links=[]
for f in [E/'REPORT.md',R/'docs/runtime-optimization/designs/1017-cpu-lookup-overlap.md']:
 for url in re.findall(r'\]\(([^)]+)\)',f.read_text()):
  if '://'in url or url.startswith('#'):continue
  target=(f.parent/url.split('#')[0]).resolve();assert target.exists(),(f,url);links.append(url)
out['report_design_links_checked']=len(links);subprocess.run(['git','diff','--check'],cwd=R,check=True)
out['diff_check']='passed';out['default']='OFF, explicit startup opt-in only';out['scope_limits']=['Other models/backends, cache/staged MoE, CPU spill, media, parallel>1 and larger contexts not certified.','No full GPU critical-path proof or actual GPU fault/OOM/cancellation recovery.','Failed/cancelled initial attempts preserved; no implementation commit/default promotion.','Private library only; no production binary replacement.'];out['complete']=True
(P/'review-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
