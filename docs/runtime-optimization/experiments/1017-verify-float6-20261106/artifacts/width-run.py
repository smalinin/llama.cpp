from pathlib import Path
import os,sys,json,subprocess,signal,threading,time,traceback,hashlib,shutil
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save(f,j):q=f.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(f)
def main():
 cfg=json.loads((P/'config.json').read_text());build=json.loads((P/'width-build.json').read_text());rb=json.loads((P/'build.json').read_text());build.update(private_library=rb['private_library'],private_library_sha256=rb['private_llama_sha256']);assert json.loads((P/'admission.json').read_text())['admit'];assert P.parent.name=='1017-verify-float6-20261106';assert not(P/'width-results.json').exists();assert cfg['command'][cfg['command'].index('--ctx-size')+1]=='92160';args=[];v=cfg['command'][1:];i=0
 while i<len(v):
  key=v[i]
  if key in ['--host','--port','--alias']:i+=2;continue
  args.append(key)
  if key in ['--no-warmup','--no-mmproj-auto']:i+=1
  else:args.append(v[i+1]);i+=2
 args+=['--no-spec-draft-backend-sampling'];j={'complete':False,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'build':build,'config':cfg,'source_hashes':{x:sha(R/x)for x in cfg['source_files']},'input_sha256':sha(P/'inputs.json'),'scope':'Uninstrumented OFF/ON raw-logit target width cost with identical fixed-prefix/input/state checks. Phase B float6, MoE5+1 explicit1 in both arms. Separate source/operator evidence only. This cohort adds no counter interposer. Prior raw screen reused byte-identical harness. Same private library, prefilling deferral explicit0/default OFF, same Q3/Q8KV/DSpark registration/ctx92160/TopK2048/full GPU placement. Query widths1..6, identical prefix/fixed input tokens/target snapshots. Two fresh processes: both without interposer, callback, CUPTI or backtrace. This is not server backend sampling or total native TG.','arms':[]};persist=lambda:save(P/'width-results.json',j);persist();refs=None
 for label in ['off','on']:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert sha(build['private_library'])==build['private_library_sha256'];assert sha(P/'width-harness')==build['harness_sha256'];assert sha(P/'width.cpp')==build['source_sha256']
  for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
  d=P/('width-'+label);d.mkdir();env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LD_LIBRARY_PATH=str(Path(build['private_library']).parent)+':'+str(R/'build-aliceai/bin'),LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSV41_VERIFY_MOE_WIDE='1',LLAMA_DSV41_VERIFY_FLOAT_WIDE=('1'if label=='on'else'0'));env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None)
  cmd=[str(P/'width-harness'),str(d),str(P/'inputs.json'),*args];a={'label':label,'complete':False,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','LD_'))}};j['arms'].append(a);persist();proc=None;mon=None;stop=threading.Event();breach=[];guard=None
  try:
   with(d/'server.log').open('wb')as log:
    proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);a['pid']=proc.pid;mon=MemoryMonitor(proc.pid);persist()
    def watch():
     while not stop.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for ph in ['load','inference']for x in mon.stats[ph].values()if isinstance(x,dict))
      if bad:
       breach.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    guard=threading.Thread(target=watch,daemon=True);guard.start();started=time.monotonic();ready=False
    while proc.poll()is None:
     txt=(d/'server.log').read_text(errors='replace')
     if not ready and '\nREADY\n'in txt:
      a['ready_seconds']=time.monotonic()-started;a['maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert a['maps']and all(str(Path(build['private_library']).parent)in x for x in a['maps']);mon.inference();ready=True;persist();print('READY_WIDTH',label,flush=True)
     assert time.monotonic()-started<2400;time.sleep(1)
    a['exit_code']=proc.returncode;assert proc.returncode==0,(label,'harness exit',proc.returncode)
   x=json.loads((d/'results.json').read_text());assert x['complete']and len(x['cases'])==2;assert not x['instrumented'];assert all(r['first_row_max_abs_vs_width1']==0 for c in x['cases']for r in c['rows']);assert all(c['same_width_logits_state_exact']and c['base_restore_exact']and len(c['rows'])==78 for c in x['cases'])
   keys={(c['prefix'],r['pass'],r['queries'],r['rep']):(r['logits_sha256'],r.get('target_state_sha256'))for c in x['cases']for r in c['rows']}
   keys.update({(c['prefix'],'base'):(c['base_state_sha256'],c['fixed_proposal_ids'])for c in x['cases']})
   if refs is None:refs=keys
   else:assert refs==keys,'OFF/ON fixed prefix/logits/state mismatch'
   a['complete']=True;a['observations']=sum(len(c['rows'])for c in x['cases']);a['exact_to_off']=True
  except Exception as ex:a['error']=repr(ex);a['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();a['forced_kill']=True
   stop.set()
   if guard:guard.join(timeout=3)
   if mon:a['memory']=mon.finish()
   a['floor_breach']=breach;a['exit_code']=proc.returncode if proc else None;txt=(d/'server.log').read_text(errors='replace');a['warnings']=[x for x in txt.splitlines()if any(t in x.lower()for t in ['does not match expectation','out of memory','failed to allocate','cuda error'])];persist()
  assert not breach and not a['warnings']and not a.get('forced_kill');print('WIDTH_ARM_DONE',label,flush=True)
 for f,h in j['source_hashes'].items():assert sha(R/f)==h,f
 for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
 for f in cfg['model_stats']:
  s=Path(f['path']).stat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==tuple(f[k]for k in ['device','inode','bytes','mtime_ns'])
 j['all_outputs_state_exact']=True;j['complete']=True;persist();print('WIDTH_BOTH_COMPLETE',flush=True)
if __name__=='__main__':main()
