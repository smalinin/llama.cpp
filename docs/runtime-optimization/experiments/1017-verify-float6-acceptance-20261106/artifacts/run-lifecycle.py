from pathlib import Path
import os,sys,json,subprocess,threading,time,signal,traceback,hashlib,re
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent;assert P.parent.name=='1017-verify-float6-acceptance-20261106'
B=R/'docs/runtime-optimization/experiments/1017-verify-float6-20261106/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest()
def save(p,j):tmp=p.with_suffix('.new');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(p)
def main():
 cfg=json.loads((P/'config.json').read_text());rb=json.loads((P/'runtime-build.json').read_text());build=json.loads((P/'build.json').read_text());assert json.loads((B/'width-results.json').read_text())['all_outputs_state_exact'];assert json.loads((P/'prior-admission.json').read_text())['admit'];assert not(P/'lifecycle-results.json').exists();candidate=B/'candidate/libllama.so.0';assert sha(candidate)==rb['private_llama_sha256']
 for name,key in [('lifecycle.cpp','source_sha256'),('lifecycle-harness','harness_sha256'),('counts.so','counter_sha256'),('fault.so','fault_sha256'),('inputs.json','inputs_sha256')]:assert sha(P/name)==build[key],name
 args=[];v=cfg['command'][1:];i=0
 while i<len(v):
  key=v[i]
  if key in ['--host','--port','--alias']:i+=2;continue
  args.append(key)
  if key in ['--no-warmup','--no-mmproj-auto']:i+=1
  else:args.append(v[i+1]);i+=2
 args+=['--no-spec-draft-backend-sampling'];args[args.index('--spec-draft-p-min')+1]='0';args[args.index('--spec-draft-n-max')+1]='5'
 j=dict(complete=False,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),config=cfg,build=build,runtime=rb,arms=[],scope='Fresh OFF/ON on private3bb55e36, MoE1/prefilldeferral0, same Q3/Q8/context92160/UUIDs/placement. Reuse original full precision sequence plus six-query backend counters and NLL126/6 added after original fault/retry.14groups/two88K corpora; exact controls including first8K, taps/state/restore/rollback/draft/NLL4/NLL6 and two seed failure/retries per arm. Counters are diagnostic; no native timing, callback or CUPTI. No runtime/default or production-module change.')
 persist=lambda:save(P/'lifecycle-results.json',j);persist();reference=None;ref_first={}
 for label,flag in [('off','0'),('on','1')]:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[x['uuid']for x in cfg['new_gpu_map']]
  for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
  for f,h in cfg['source_files'].items():assert sha(R/f)==h,f
  for m in cfg['model_stats']:
   st=Path(m['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==tuple(m[k]for k in ['device','inode','bytes','mtime_ns'])
  env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LLAMA_DSV41_VERIFY_MOE_WIDE='1',LLAMA_DSV41_VERIFY_FLOAT_WIDE=flag,LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LD_LIBRARY_PATH=str(candidate.parent)+':'+str(R/'build-aliceai/bin'),LD_PRELOAD=str(P/'counts.so')+':'+str(P/'fault.so'),DSV41_TRACE_FILE=str(P/('lifecycle-'+label+'-faults.json')));env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None)
  d=P/('lifecycle-'+label);d.mkdir();cmd=[str(P/'lifecycle-harness'),str(d),str(P/'inputs.json'),*args];a=dict(label=label,flag=flag,complete=False,command=cmd,environment={k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','LD_','DSV'))});j['arms'].append(a);persist();proc=None;mon=None;halt=threading.Event();floor=[];watcher=None
  try:
   with(d/'server.log').open('wb')as log:
    proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['pid']=proc.pid;persist();mon=MemoryMonitor(proc.pid)
    def watch():
     while not halt.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for phase in ['load','inference']for x in mon.stats[phase].values()if isinstance(x,dict))
      if bad:
       floor.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    watcher=threading.Thread(target=watch,daemon=True);watcher.start();start=time.monotonic();ready=False;last=None
    while proc.poll()is None:
     txt=(d/'server.log').read_text(errors='replace')
     if not ready and '\nREADY\n'in txt:
      a['maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert a['maps']and all(str(candidate.parent)in x for x in a['maps']);ready=True;mon.inference();a['ready_seconds']=time.monotonic()-start;persist();print('READY_LIFECYCLE',label,flush=True)
     f=d/'results.json'
     if f.exists():
      try:q=json.loads(f.read_text())
      except json.JSONDecodeError:q=None
      if q:
       progress=(q.get('progress'),len(q['cases']))
       if progress!=last:print('PROGRESS',label,progress,flush=True);last=progress
       for corpus,row in q.get('first8K_controls',{}).items():
        if reference is not None and ref_first[corpus]!=row:a['early_mismatch']={'first8K':corpus};persist();raise AssertionError(('first8K mismatch before candidate branch',corpus))
       for row in q['cases']:
        key=(row['corpus'],row['key'])
        if row['key']=='rollback':assert all(x['target_replay']and x['draft_replay']for x in row['result']),('rollback self replay',key)
        if reference is not None and reference[key]!=row['result']:
         save(P/'first-difference.json',dict(corpus=key[0],key=key[1],off=reference[key],on=row['result']));a['early_mismatch']={'corpus':key[0],'key':key[1]};persist();raise AssertionError(('paired lifecycle mismatch',key))
     assert time.monotonic()-start<3600;time.sleep(1)
    a['exit_code']=proc.returncode;assert proc.returncode==0,('lifecycle exit',label,proc.returncode)
   result=json.loads((d/'results.json').read_text());assert result['complete']and len(result['cases'])==14 and len(result['faults'])==2 and len(result['dispatch'])==56
   rows={(x['corpus'],x['key']):x['result']for x in result['cases']};assert len(rows)==14
   if reference is None:reference=rows;ref_first=result['first8K_controls']
   else:assert rows==reference
   faults=json.loads((P/('lifecycle-'+label+'-faults.json')).read_text());assert faults['injected_failures']==2
   a.update(complete=True,groups_exact=14,faults=faults)
  except Exception as ex:a['error']=repr(ex);a['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();a['forced_kill']=True
   halt.set()
   if watcher:watcher.join(timeout=3)
   if mon:a['memory']=mon.finish()
   a['floor_breach']=floor;a['exit_code']=proc.returncode if proc else None;txt=(d/'server.log').read_text(errors='replace');a['warnings']=[x for x in txt.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',x,re.I)];persist()
  assert not floor and not a['warnings']and not a.get('forced_kill');print('LIFECYCLE_ARM_DONE',label,flush=True)
 off=json.loads((P/'lifecycle-off/results.json').read_text());on=json.loads((P/'lifecycle-on/results.json').read_text());pairs=[]
 for a,b in zip(off['dispatch'],on['dispatch']):
  assert all(a[k]==b[k]for k in ['corpus','position','tokens']);ca,cb=a['counts'],b['counts'];assert ca['moe_matmul']==cb['moe_matmul']==240 and ca['quant_matmul']==cb['quant_matmul'];assert cb['float_matmul']<ca['float_matmul'];pairs.append(dict(corpus=a['corpus'],position=a['position'],off=ca,on=cb))
 assert len(pairs)==56;save(P/'dispatch-summary.json',dict(complete=True,matching_six_query_pairs=56,pairs=pairs,scope='Logical GGML backend graph nodes and API launch/copy/sync counters, not GPU kernel durations or total kernels inside replayed CUDA Graphs.'))
 for f,h in cfg['source_files'].items():assert sha(R/f)==h,f
 for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
 j.update(all_groups_exact=True,all_first8K_exact=True,dispatch_reduction_verified=True,complete=True);persist();print('LIFECYCLE_COMPLETE',flush=True)
if __name__=='__main__':main()
