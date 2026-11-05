from pathlib import Path
import json,hashlib,gzip,subprocess,statistics,collections,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105';P=E/'artifacts';D=P/'shared-local'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(D/'counts-results.json');assert r['complete']and r['requests_complete']and r['exit_code']==0 and not r['floor_breach']and not r['warnings']and not r.get('forced_kill')and len(r['requests'])==5
for f,h in r['config']['source_library_hashes'].items():assert sha(Path(f))==h,f
for f,h in r['config']['source_files'].items():assert sha(R/f)==h,f
for x in r['config']['model_stats']:
 s=Path(x['path']).stat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==tuple(x[k]for k in ['device','inode','bytes','mtime_ns'])
assert 'CUDA_SCALE_LAUNCH_QUEUES'not in r['environment'] and 'GGML_SCHED_CPU_LOOKUP_EARLY'not in r['environment']
base=read(P/'launch-attribution/counts-results.json');expected=base['command'][:];expected[expected.index('--override-tensor')+1]+=r',blk\.6\.ffn_(gate|up)_shexp\.weight=CUDA1,blk\.29\.ffn_(gate|up)_shexp\.weight=CUDA4'
for option in ['--port','--alias']:expected[expected.index(option)+1]=r['command'][r['command'].index(option)+1]
assert expected==r['command'];assert r['environment']['LD_LIBRARY_PATH']==base['environment']['LD_LIBRARY_PATH']
ref={x['prefix']:(x['tokens_sha256'],x['response']['timings']['draft_n'],x['response']['timings']['draft_n_accepted'])for x in base['requests']};exact=all((x['tokens_sha256'],x['response']['timings']['draft_n'],x['response']['timings']['draft_n_accepted'])==ref[x['prefix']]for x in r['requests'])
controls=lambda j:statistics.mean(x['pp']for x in j['requests']if x['kind']in ['pre','post'])
gain=100*(controls(r)/controls(base)-1)
# Analysis source is preserved with each phase; no server rerun.
s=(P/'launch-attribution/analyze.py').read_text().replace('/artifacts/launch-attribution','/artifacts/shared-local');(D/'analyze.py').write_text(s);subprocess.run(['python3',str(D/'analyze.py')],cwd=R,check=True)
s=(P/'analyze.py').read_text().replace("P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts'","P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts/shared-local'").replace("P/'summary.json'","P/'host-summary.json'");(D/'analyze-host.py').write_text(s);subprocess.run(['python3',str(D/'analyze-host.py')],cwd=R,check=True)
transitions={}
for name in ['launch-attribution','shared-local']:
 rows=[json.loads(l)for l in(P/name/'scopes.jsonl').read_text().splitlines()];c=collections.Counter()
 for row in rows:
  if row['kind']!='ubatch':continue
  calls=row['calls']
  for i,x in enumerate(calls):
   if x['kind']!='event_wait'or x['backend']==x.get('event_device')or x['offset']!=0x46e91:continue
   nxt=next((y for y in calls[i+1:]if y['kind']=='graph_compute'),None)
   c[(x['event_device'],x['backend'],nxt['first']['name']if nxt else'',nxt['first']['op']if nxt else'',nxt['nodes']if nxt else 0)]+=1
 transitions[name]=[{'source':k[0],'dest':k[1],'first_name':k[2],'first_op':k[3],'nodes':k[4],'count':v}for k,v in c.items()]
# The absent backward gate paths confirm that the candidate took effect; KV writes remain protected.
old=sum(x['count']for x in transitions['launch-attribution']if(x['source'],x['dest'])in [('CUDA1','CUDA0'),('CUDA4','CUDA3')]);new=sum(x['count']for x in transitions['shared-local']if(x['source'],x['dest'])in [('CUDA1','CUDA0'),('CUDA4','CUDA3')])
free={ph:{k:v['min_free']/2**20 for k,v in r['memory'][ph].items()if isinstance(v,dict)}for ph in ['load','inference']};assert min(v for ph in free.values()for v in ph.values())>=512
j={'complete':True,'native_requests':5,'native_pp_baseline_control_mean':controls(base),'native_pp_candidate_control_mean':controls(r),'gain_percent':gain,'control_values':[(x['kind'],x['pp'],x['tg'])for x in r['requests']],'all_outputs_counters_exact_to_baseline':exact,'reverse_transition_before':old,'reverse_transition_after':new,'transitions':transitions,'runtime_source_and_libraries_unchanged':True,'target_shard_metadata_unchanged':10,'no_new_default_or_preset':True,'min_free_mib':free,'exit_code':0,'floor_breach':[],'warnings':[],'quality_scope':'16-token greedy IDs and draft counters only. Full raw-logit/state/rollback/NLL gates and proper fresh-process speed bracketing are required before adopting a positive screen. No new source/runtime feature.'};save(D/'checks.json',j)
raw=(D/'scopes.jsonl').read_bytes();packed=gzip.compress(raw,mtime=0);(D/'scopes.jsonl.gz').write_bytes(packed);assert gzip.decompress(packed)==raw;save(D/'payload.json',{'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'gzip_bytes':len(packed),'gzip_sha256':hashlib.sha256(packed).hexdigest()})
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
shutil.copy2('/tmp/close-residual-all.py',P/'close-residual-all.py');shutil.copy2('/tmp/write-residual-report.py',P/'write-residual-report.py');print('SHARED_LOCAL',json.dumps({k:j[k]for k in ['gain_percent','all_outputs_counters_exact_to_baseline','reverse_transition_before','reverse_transition_after']}))
