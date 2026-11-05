from pathlib import Path
import json,os,sys,time,subprocess,signal,threading,hashlib,traceback,re
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-forensic-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';S=R/'docs/runtime-optimization/experiments/1017-verify-shared-prefix-20261105/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):q=p.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def main():
 old=json.loads((O/'precision-results.json').read_text());cfg=old['config'];args=old['arms'][0]['command'][3:];build=json.loads((O/'build.json').read_text());assert json.loads((O/'admission.json').read_text())['admit'];private=O/'candidate/libllama.so.0';dest=P/'results.json';assert not dest.exists();assert not(P/'bundle').exists();(P/'bundle').mkdir();reference=None
 canonical=next(x['result']['first8K']for x in json.loads((R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts/precision-0/results.json').read_text())['cases']if x['corpus']=='technical-prose'and x['key']=='prefix90112')
 actual_draft=sha(Path(cfg['draft']['path']));assert actual_draft==cfg['draft']['sha256'];draft_stat=Path(cfg['draft']['path']).stat()
 j={'complete':False,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'protocol':'../PROTOCOL.md','config':cfg,'private_llama_sha256':sha(private),'harness_sha256':sha(P/'prefill-forensic-harness'),'harness_source_sha256':sha(P/'prefill-forensic.cpp'),'fault_sha256':sha(P/'fault.so'),'input_sha256':sha(P/'inputs.json'),'actual_draft_sha256':actual_draft,'draft_stat':{'device':draft_stat.st_dev,'inode':draft_stat.st_ino,'bytes':draft_stat.st_size,'mtime_ns':draft_stat.st_mtime_ns},'admission_reuse':{'manifest':str(O/'admission.json'),'sha256':sha(O/'admission.json'),'scope':'Unchanged private runtime/context/output6/model placement, existing safe joint admission. Only host snapshot retention/raw dumping/metadata serialization added after first8K apply. Actual512MiB floor monitored.'},'arms':[]};save(dest,j)
 for index in range(3):
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert sha(private)==j['private_llama_sha256'];assert sha(P/'prefill-forensic-harness')==j['harness_sha256'];assert sha(P/'prefill-forensic.cpp')==j['harness_source_sha256'];assert sha(P/'fault.so')==j['fault_sha256'];assert all(sha(R/x['source'])==x['source_sha256']for x in build['compile']);assert all(sha(f)==h for f,h in cfg['source_library_hashes'].items());assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[x['uuid']for x in cfg['new_gpu_map']]
  for m in cfg['model_stats']:
   st=Path(m['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(m['device'],m['inode'],m['bytes'],m['mtime_ns'])
  st=Path(cfg['draft']['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(draft_stat.st_dev,draft_stat.st_ino,draft_stat.st_size,draft_stat.st_mtime_ns)
  env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSV41_VERIFY_MOE_WIDE='0',LD_LIBRARY_PATH=str(O/'candidate')+':'+str(R/'build-aliceai/bin'),LD_PRELOAD=str(P/'fault.so'),DSV41_TRACE_FILE=str(P/f'faults-{index}.json'))
  d=P/f'arm-{index}';d.mkdir();command=[str(P/'prefill-forensic-harness'),str(d),str(P/'inputs.json'),str(P/'bundle'),*args];arm={'index':index,'flag':'0','complete':False,'command':command,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))}};j['arms'].append(arm);save(dest,j);proc=None;mon=None;watch=None;halt=threading.Event();floor=[]
  try:
   with(d/'server.log').open('wb')as log:
    proc=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);arm['pid']=proc.pid;save(dest,j);mon=MemoryMonitor(proc.pid)
    def monitor():
     while not halt.wait(.2):
      with mon.lock:bad=any(x.get('min_free',512*2**20)<512*2**20 for phase in ['load','inference']for x in mon.stats[phase].values()if isinstance(x,dict))
      if bad:
       floor.append(time.time())
       if proc.poll()is None:proc.send_signal(signal.SIGTERM)
       return
    watch=threading.Thread(target=monitor,daemon=True);watch.start();start=time.monotonic();ready=False
    while proc.poll()is None:
     txt=(d/'server.log').read_text(errors='replace')
     if not ready and '\nREADY\n'in txt:
      maps=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert maps and all(str(O/'candidate')in x for x in maps);arm['loaded_llama_maps']=maps;ready=True;mon.inference();save(dest,j);print('READY_FORENSIC',index,flush=True)
     assert time.monotonic()-start<2400;time.sleep(1)
   assert proc.returncode==0,('exit',proc.returncode);res=json.loads((d/'results.json').read_text());assert res['complete']and len(res['cases'])==3;arm['result_sha256']=sha(d/'results.json')
   if reference is None:reference=res['cases'][0]['value'];j['first_observed_reference']=reference
   arm['reference_exact']=[x['value']==reference for x in res['cases']];arm['canonical_exact']=[x['value']==canonical for x in res['cases']];arm['tap_raw_matches_original']=[x['tap_raw_matches_original']for x in res['cases']];arm['metadata_consistent']=[x['metadata_consistent']for x in res['cases']];fault=json.loads((P/f'faults-{index}.json').read_text());assert fault['injected_failures']==0;arm['faults']=fault;arm['complete']=True;print('ARM_DONE',index,'reference',arm['reference_exact'],'canonical',arm['canonical_exact'],flush=True)
  except Exception as ex:arm['error']=repr(ex);arm['traceback']=traceback.format_exc();raise
  finally:
   if proc and proc.poll()is None:
    proc.send_signal(signal.SIGTERM)
    try:proc.wait(timeout=45)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();arm['forced_kill']=True
   halt.set()
   if watch:watch.join(timeout=3)
   if mon:arm['memory']=mon.finish()
   arm['floor_breach']=floor;arm['exit_code']=proc.returncode if proc else None;txt=(d/'server.log').read_text(errors='replace');arm['critical_warnings']=[x for x in txt.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',x,re.I)];save(dest,j)
  assert not floor and not arm.get('forced_kill')and not arm['critical_warnings']
 j['all_reference_exact']=all(all(a['reference_exact'])for a in j['arms']);j['all_canonical_exact']=all(all(a['canonical_exact'])for a in j['arms']);j['complete']=True;save(dest,j);print('PREFILL_FORENSIC_COMPLETE',j['all_reference_exact'],j['all_canonical_exact'],flush=True)
if __name__=='__main__':main()
