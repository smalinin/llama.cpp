from pathlib import Path
import json,collections,hashlib,shutil,subprocess,xml.etree.ElementTree as ET
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-maya-telemetry-20261105';P=E/'artifacts'
read=lambda p:json.loads(p.read_text());dump=lambda p,j:p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
jm=read(P/'results.json');jo=read(P/'ours-results.json');cfg=read(P/'config.json');assert jo['complete'] and jo['arms'][0]['exit_code']==0
for src,dest in [('/tmp/analyze-dsv41-timeline.py','analyze.py'),('/tmp/dsv41-extra-telemetry.py','scopes.py'),('/tmp/dsv41-svg-timeline.py','plot-svg.py'),('/tmp/ours-scopes8k.json','ours/scopes8k.json'),('/tmp/ours-scopes98k.json','ours/scopes98k.json')]:shutil.copy2(src,P/dest)
ET.parse(P/'prefill8k-timeline.svg')
summary={'status':'analysis_complete_with_recovered_Maya98K_limitations','all_clean_controlled_arms_complete':False,'ours_controller_complete':True,'original_Maya_controller_complete':False,'native_reference':'../1017-maya-newhost-20261105/REPORT.md','controls':[],'profiles':[],'our_scopes':{},'memory_floor_mib':{}}
for engine,j in [('maya',jm),('ours',jo)]:
 arm=j['arms'][0];inf=arm['memory']['inference'];summary['memory_floor_mib'][engine]=min(v['min_free']for v in inf.values()if isinstance(v,dict)and'min_free'in v)/2**20
 for r in arm['requests']:
  b,z=r['before'],r['after'];t=r['timings'];summary['controls'].append({'engine':engine,'prefix':r['n'],'kind':r['kind'],'PP':r['n']*1000/t['prompt_ms'],'TG_127_intervals':127000/t['predicted_ms'],'cpu_seconds':z['cpu_s']-b['cpu_s'],'wall_s':r['wall_s'],'major_faults':z['major_faults']-b['major_faults'],'minor_faults':z['minor_faults']-b['minor_faults'],'read_bytes':z['io']['read_bytes']-b['io']['read_bytes'],'write_bytes':z['io']['write_bytes']-b['io']['write_bytes'],'output_sha256':r['tokens_sha256'],'acceptance':t['draft_n_accepted']/t['draft_n']if t['draft_n']else None})
 for n in ['8k','98k']:
  a=read(P/engine/('analysis'+n+'.json'));summary['profiles'].append({'engine':engine,'prefix':n,'kernels':a['kernel_records'],'runtime_api_records':a['runtime_records'],'memcpy_records':a['memcpy_records'],'OSRT_available':a['OSRT_available'],'diagnostics':[{'text':t,'count':v}for t,v in collections.Counter(t for _,t in a['diagnostics']).items()],'phases':a['phases']})
  if engine=='ours':
   sc=read(P/engine/('scopes'+n+'.json'));summary['our_scopes'][n]=[{'phase':x['phase'],'summary':x['scopes_summary']}for x in sc['phases']]
# Keep the failed controller record intact. Recovery is a separate provenance record.
rec=read(P/'recovery.json');rec['analysis_state']='Both Maya traces exported and analysed; 98K recovered DONE retained separately. Ours corrected controller completed9requests,2captures,exit0.';rec['remaining_limits']='Maya98K has no post-control or phase CPU/I/O counters; diagnostics include stop-recording timeout. Original runner manifest remains incomplete.';dump(P/'recovery.json',rec)
dump(P/'summary.json',summary)
checks={}
for group in ['source_library_hashes','source_aux_hashes']:
 checks[group]={}
 for name,expected in cfg[group].items():
  h=hashlib.sha256()
  with Path(name).open('rb')as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  value=h.hexdigest();checks[group][name]={'expected':expected,'actual':value,'match':value==expected}
checks['source_files']={name:hashlib.sha256((R/name).read_bytes()).hexdigest()==expected for name,expected in cfg['source_files'].items()}
checks['head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();checks['branch']=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip();checks['runtime_source_diff']=subprocess.check_output(['git','diff','--','src','ggml/src','common','tools/server'],cwd=R,text=True)
assert all(v['match']for group in ['source_library_hashes','source_aux_hashes']for v in checks[group].values());assert all(checks['source_files'].values());assert not checks['runtime_source_diff'];dump(P/'after-checks.json',checks)
print('summary and identity checks saved; floors',summary['memory_floor_mib'])
