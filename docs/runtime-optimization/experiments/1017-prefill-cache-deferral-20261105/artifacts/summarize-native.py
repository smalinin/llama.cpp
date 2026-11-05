from pathlib import Path
import json,statistics,hashlib,subprocess,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105';P=E/'artifacts'
j=json.loads((P/'native-results.json').read_text());assert j['complete']and j['all_outputs_counters_exact'];assert [a['flag']for a in j['arms']]==['0','1','0']
rows=[]
for n in [512,8192,90112]:
 byarm={a['label']:[r for r in a['requests']if r['prefix']==n and r['kind']=='measured']for a in j['arms']};assert all(len(v)==2 for v in byarm.values())
 baseline=byarm['off-pre']+byarm['off-post'];on=byarm['on'];r={'prefix':n,'samples':byarm}
 for m in ['pp','tg']:
  offmean=statistics.mean(x[m]for x in baseline);onmean=statistics.mean(x[m]for x in on)
  r[m]={'off_mean':offmean,'on_mean':onmean,'delta_percent':100*(onmean/offmean-1),'off_range':[min(x[m]for x in baseline),max(x[m]for x in baseline)],'on_range':[min(x[m]for x in on),max(x[m]for x in on)],'off_pre_mean':statistics.mean(x[m]for x in byarm['off-pre']),'off_post_mean':statistics.mean(x[m]for x in byarm['off-post'])};r[m]['control_drift_percent']=100*(r[m]['off_post_mean']/r[m]['off_pre_mean']-1)
 rows.append(r)
mem={}
for a in j['arms']:
 assert a['exit_code']==0 and not a['warnings']and not a['floor_breach']and not a.get('forced_kill')
 mem[a['label']]={ph:{str(k):v['min_free']/2**20 for k,v in a['memory'][ph].items()if isinstance(v,dict)}for ph in ['load','inference']}
 assert min(v for phase in mem[a['label']].values()for v in phase.values())>=512
reserve={mode:json.loads((P/f'context-92160/{mode}.json').read_text())for mode in ['off','on']};deltas=[]
for off,on in zip(reserve['off']['memory'],reserve['on']['memory']):
 assert off['device']==on['device'];keys=['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute'];deltas.append({'device':off['device'],**{k:(on[k]-off[k])/2**20 for k in keys},'total_delta_mib':sum(on[k]-off[k]for k in keys)/2**20})
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest()
assert sha(P/'candidate/libllama.so.0')==j['candidate_sha256'];assert subprocess.check_output(['git','diff','--','src'],cwd=R,text=True)==(P/'runtime.patch').read_text()
for f,h in j['config']['source_library_hashes'].items():assert sha(f)==h,f
for f,h in j['config']['source_files'].items():assert sha(R/f)==h,f
for f in j['config']['model_stats']:
 st=Path(f['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(f[k]for k in ['device','inode','bytes','mtime_ns'])
out={'complete':True,'native_complete':True,'precision_exact':True,'actual_owner_and_flush_complete':True,'context':92160,'long_prefix':90112,'gpu_floor_mib':512,'default':'OFF; only explicit1 opts in','comparison':'ON mean of two measured repetitions vs four measured OFF repetitions bracketing ON; warmups excluded. Short screen, not a statistical confidence interval.','rows':rows,'sampled_min_free_mib':mem,'reserve_deltas_mib':deltas,'source_libraries_unchanged':True,'target_shards_stat_unchanged':10,'private_library_sha256':j['candidate_sha256']}
(P/'native-summary.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2('/tmp/summarize-deferral-native.py',P/'summarize-native.py')
for r in rows:print(r['prefix'],{m:{k:round(v,3)if isinstance(v,(float,int))else v for k,v in r[m].items()if k not in ['off_range','on_range']}for m in ['pp','tg']})
print('minimum',min(v for a in mem.values()for ph in a.values()for v in ph.values()));print('reserve delta',deltas)
