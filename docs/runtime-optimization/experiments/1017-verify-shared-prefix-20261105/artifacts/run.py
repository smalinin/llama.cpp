from pathlib import Path
import json,os,sys,time,subprocess,signal,threading,hashlib,traceback,re
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-shared-prefix-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';sys.path.insert(0,str(P));from memory_monitor import MemoryMonitor

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):q=p.with_suffix('.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def differences(a,b,path=''):
 if isinstance(a,dict)and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):yield from differences(a.get(k),b.get(k),path+'.'+k)
 elif isinstance(a,list)and isinstance(b,list):
  if len(a)!=len(b):yield {'path':path+'.length','reference':len(a),'actual':len(b)}
  for i,(x,y)in enumerate(zip(a,b)):yield from differences(x,y,path+f'[{i}]')
 elif a!=b:yield {'path':path,'reference':a,'actual':b}
def main():
 old=json.loads((O/'precision-results.json').read_text());cfg=old['config'];args=old['arms'][0]['command'][3:];build=json.loads((O/'build.json').read_text());admit=json.loads((O/'admission.json').read_text());assert admit['admit'];private=O/'candidate/libllama.so.0';dest=P/'results.json';assert not dest.exists();assert not(P/'bundle').exists();(P/'bundle').mkdir()
 j={'complete':False,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'protocol':'../PROTOCOL.md','config':cfg,'private_llama_sha256':sha(private),'source_sha256':{c['source']:sha(R/c['source'])for c in build['compile']},'harness_sha256':sha(P/'shared-prefix-harness'),'harness_source_sha256':sha(P/'shared-prefix.cpp'),'counter_sha256':sha(P/'counts.so'),'input_sha256':sha(P/'inputs.json'),'admission_reuse':{'manifest':str(O/'admission.json'),'sha256':sha(O/'admission.json'),'scope':'Same private source/library, context92160, args/placement/output6 and previously admitted reserve. No new runtime tensors/kernel/API/class changes; only host blob serialization. Actual>=512MiB monitor required.'},'arms':[]};save(dest,j);reference=None
 for index,flag in enumerate(['0','1','0']):
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert sha(private)==j['private_llama_sha256'];assert sha(P/'shared-prefix-harness')==j['harness_sha256'];assert sha(P/'shared-prefix.cpp')==j['harness_source_sha256'];assert sha(P/'counts.so')==j['counter_sha256'];assert all(sha(R/x['source'])==x['source_sha256']for x in build['compile']);assert all(sha(f)==h for f,h in cfg['source_library_hashes'].items())
  actual_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();assert actual_uuid==[x['uuid']for x in cfg['new_gpu_map']]
  for m in cfg['model_stats']:
   st=Path(m['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(m['device'],m['inode'],m['bytes'],m['mtime_ns'])
  env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.update(LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSV41_VERIFY_MOE_WIDE=flag,LD_LIBRARY_PATH=str(O/'candidate')+':'+str(R/'build-aliceai/bin'),LD_PRELOAD=str(P/'counts.so'))
  mode='generate'if index==0 else'restore';d=P/('arm-'+str(index)+'-'+flag);d.mkdir();command=[str(P/'shared-prefix-harness'),str(d),str(P/'inputs.json'),str(P/'bundle'),mode,*args];arm={'index':index,'flag':flag,'mode':mode,'complete':False,'command':command,'environment':{k:v for k,v in env.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','LD_'))}};j['arms'].append(arm);save(dest,j);proc=None;mon=None;watch=None;halt=threading.Event();floor=[]
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
      maps=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert maps and all(str(O/'candidate')in x for x in maps);arm['loaded_llama_maps']=maps;ready=True;mon.inference();save(dest,j);print('READY_SHARED',index,flag,flush=True)
     assert time.monotonic()-start<2400;time.sleep(1)
   assert proc.returncode==0,('exit',proc.returncode);res=json.loads((d/'results.json').read_text());assert res['complete']and len(res['cases'])==39 and len(res['draft_checks'])==6;arm['internal_exact']=res['internal_exact'];arm['seed_sha256']=sha(P/'bundle/seed.json');arm['result_sha256']=sha(d/'results.json');arm['n6_moe_counts']=[r['counts']['moe_matmul']for r in res['cases']if r['value']['n']==6];arm['expected_n6_moe_count']=240 if flag=='1'else 720;assert all(n==arm['expected_n6_moe_count']for n in arm['n6_moe_counts']),arm['n6_moe_counts']
   values={'seed':res['seed'],'draft_checks':res['draft_checks'],'cases':[{'key':r['key'],'value':r['value']}for r in res['cases']]}
   if reference is None:reference=values;arm['reference']=True;arm['differences']=[]
   else:arm['differences']=list(differences(reference,values));arm['cross_arm_exact']=not arm['differences']
   arm['complete']=True;print('ARM_DONE',index,flag,'internal',arm['internal_exact'],'differences',len(arm['differences']),flush=True)
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
 j['all_internal_exact']=all(a['internal_exact']for a in j['arms']);j['all_cross_arm_exact']=all(not a['differences']for a in j['arms']);j['complete']=True;save(dest,j);print('SHARED_PREFIX_COMPLETE',j['all_internal_exact'],j['all_cross_arm_exact'],flush=True)
if __name__=='__main__':main()
