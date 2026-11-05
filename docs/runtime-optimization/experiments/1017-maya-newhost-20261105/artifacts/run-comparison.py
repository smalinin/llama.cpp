from pathlib import Path
from urllib.request import Request,urlopen
import os,json,time,subprocess,signal,hashlib,shutil,threading,queue,re,traceback,statistics,sys
P=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts')
sys.path.insert(0,str(P))
from memory_monitor import MemoryMonitor

def save(path,j):
 tmp=path.with_suffix(path.suffix+'.new');tmp.write_text(json.dumps(j,indent=2)+'\n');tmp.replace(path)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def idle():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20);assert not out.strip(),('GPU busy',out)
def cpu(pid):
 a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return (int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK')
def stop(proc,maya=False):
 if proc and proc.poll() is None:
  if maya:
   try:proc.stdin.write('QUIT\n');proc.stdin.flush()
   except (BrokenPipeError,OSError):pass
  else:proc.send_signal(signal.SIGTERM)
  try:proc.wait(timeout=60)
  except subprocess.TimeoutExpired:proc.kill();proc.wait();return True
 return False

def main():
 cfg=json.loads((P/'config.json').read_text());tokens=json.loads((P/'inputs.json').read_text())['tokens'];assert not(P/'results.json').exists()
 base={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};base.update({k:cfg['environment'][k] for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES']})
 plan=[(n,kind,rep) for n,warms,reps in [(512,2,3),(8192,2,3),(98304,1,2)] for kind,rep in [('warmup',-i-1) for i in range(warms)]+[('measured',i) for i in range(reps)]]
 j={'complete':False,'cohort':'2026-11-05 new motherboard/CPU/RAM/PCIe topology; historical speed and cost values excluded','config':cfg,'plan':plan,'arms':[],'started':time.time(),'protocol':'52 full uncached requests, 256 greedy outputs. TG=255 intervals for both; first token included in PP. Same ten Q3 GGUF inodes and sidecar file, ctx100352, TopK2048, U/chunk512, CPUthreads12. Ours B8192/Q8KV/cache0/staging0/fixedGPU overrides; Maya native KV and placement/reserve1024/hot frequency profile/fresh lane calibration. K5/pmin.5/cumulative0/cooldown1. End-to-end comparison with differing arithmetic/KV/placement, not an isolated ablation or quality gate. CPU seconds include PP+TG/protocol, not decode-only. One arm order, 98K only two measured runs.'}
 def persist():save(P/'results.json',j)
 persist()
 for engine,spec in [('ours',False),('maya',False),('maya',True),('ours',True)]:
  idle()
  for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
  label=engine+('-on' if spec else '-off');d=P/label;d.mkdir(exist_ok=False);e=base.copy()
  if engine=='ours':
   e.update(cfg['environment']);cmd=cfg['command'][:];cmd[cmd.index('--alias')+1]=label
   if not spec:
    for k in ['--spec-type','--spec-draft-model','--spec-draft-n-max','--spec-draft-p-min','--gpu-layers-draft','--spec-draft-device','--spec-verify-policy','--spec-draft-override-tensor']:
     i=cmd.index(k);del cmd[i:i+2]
  else:
   shutil.copytree(P/'pack-base',d/'pack');assert not(d/'pack/lane-times.txt').exists()
   e.update(STRATA_DS4_SLOT_KEEP='0',STRATA_DS4_DSPARK_CUM='0',DS4_DSPARK_MAX='5')
   cmd=[cfg['maya_binary'],'--pack',str(d/'pack'),'--max-context','100352','--gpus','0,1,2,3,4,5','--tp','1','--threads','on','--cpu-threads','12','--reserve-mib','1024','--prefill-chunk','512','--mmap-experts','auto','--dspark','on' if spec else 'none','--dspark-conf','0.5','--vision','off','--eos','-1','--serve','--verbose']
  a={'label':label,'engine':engine,'sidecar':spec,'complete':False,'command':cmd,'environment':{k:v for k,v in e.items() if k.startswith(('CUDA_','LLAMA_','GGML_','STRATA_','DS4_','LD_'))},'requests':[]};j['arms'].append(a);persist()
  proc=None;mon=None;guard_stop=threading.Event();floor=[];watch=None;reader=None
  try:
   with(d/'server.log').open('wb') as log:
    proc=subprocess.Popen(cmd,cwd=d,env=e,stdin=subprocess.PIPE if engine=='maya' else subprocess.DEVNULL,stdout=subprocess.PIPE if engine=='maya' else log,stderr=log if engine=='maya' else subprocess.STDOUT,text=engine=='maya',bufsize=1 if engine=='maya' else -1,start_new_session=True);a['pid']=proc.pid;save(P/'active.json',{'pid':proc.pid,'label':label});mon=MemoryMonitor(proc.pid)
    def guard():
     while not guard_stop.wait(.2):
      with mon.lock:breach=any(v.get('min_free',512*1048576)<512*1048576 for phase in ['load','inference'] for v in mon.stats[phase].values() if isinstance(v,dict))
      if breach:
       floor.append(time.monotonic())
       if proc.poll() is None:proc.send_signal(signal.SIGTERM)
       return
    watch=threading.Thread(target=guard,daemon=True);watch.start()
    if engine=='ours':
     def api(path,body=None,timeout=5):
      with urlopen(Request('http://127.0.0.1:19089'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=timeout) as f:return json.load(f)
     start=time.monotonic()
     while True:
      assert proc.poll() is None,('load exit',proc.returncode)
      try:
       if api('/health').get('status')=='ok':break
      except OSError:pass
      assert time.monotonic()-start<1500;time.sleep(1)
     def request(ids,count):return api('/completion',dict(prompt=ids,n_predict=count,temperature=0,top_k=1,top_p=1,min_p=0,repeat_penalty=1,seed=1234,ignore_eos=True,cache_prompt=False,return_tokens=True,id_slot=0),1500)
    else:
     q=queue.Queue();transcript=(d/'protocol.log').open('w')
     def read_lines():
      for line in proc.stdout:transcript.write(line);transcript.flush();q.put(line.strip())
      q.put(None)
     reader=threading.Thread(target=read_lines,daemon=True);reader.start()
     def get():
      line=q.get(timeout=1500)
      if line is None or line.startswith('ERR '):raise RuntimeError(str(line))
      return line
     while not get().startswith('READY '):pass
     def request(ids,count):
      proc.stdin.write(f'GEN {count} temperature=0 top_p=1 top_k=1 min_p=0 seed=1234 dspark_conf=0.5 dspark_cum=0 '+','.join(map(str,ids))+'\n');proc.stdin.flush();out=[];stats=[]
      while True:
       line=get()
       if line.startswith('T '):out.append(int(line.split()[1]))
       elif line.startswith('STAT '):stats.append(line)
       elif line.startswith('DONE '):return {'tokens':out,'done':line,'stats':stats}
    mon.inference();print('READY',label,flush=True)
    for n,kind,rep in plan:
     request([97],1);before=cpu(proc.pid);start=time.monotonic();v=request(tokens[:n],256);wall=time.monotonic()-start;cs=cpu(proc.pid)-before
     if engine=='ours':
      t=v['timings'];assert t['prompt_n']==n and t['cache_n']==0 and t['predicted_n']==len(v['tokens'])==256,t
      pp=t['prompt_ms'];tg=t['predicted_ms'];acc=t.get('draft_n_accepted',0);off=t.get('draft_n',0)
     else:
      f=v['done'].split();assert int(f[1])==len(v['tokens'])==256 and int(f[2])==n and int(f[8])==0,v['done']
      pp=float(f[3]);tg=float(f[4]);acc=int(f[6]);off=int(f[7])
     row={'prefix':n,'kind':kind,'rep':rep,'pp':n*1000/pp,'tg':255*1000/tg,'pp_ms':pp,'tg_ms':tg,'wall_s':wall,'cpu_s':cs,'mean_cpu_cores':cs/wall,'draft_accepted':acc,'draft_offered':off,'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest(),'response':v}
     a['requests'].append(row);persist();assert not floor
     print('CASE',label,n,kind,rep,round(row['pp'],2),round(row['tg'],2),'CPU',round(cs,2),'draft',acc,off,flush=True)
    a['complete']=True
  except Exception as ex:a['error']=repr(ex);a['traceback']=traceback.format_exc();raise
  finally:
   guard_stop.set()
   if watch:watch.join(timeout=3)
   a['forced_kill']=stop(proc,engine=='maya');a['exit_code']=proc.returncode if proc else None
   if reader:reader.join(timeout=3);transcript.close()
   if mon:a['memory']=mon.finish()
   a['floor_breach']=floor;text=(d/'server.log').read_text(errors='replace');a['warnings']=[x for x in text.splitlines() if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',x,re.I)];a['placement_lines']=[x for x in text.splitlines() if re.search(r'CUDA\d+ layers|routed experts on the CPU lane|model buffer size|offloaded .*layers',x)];a['per_position']=[x for x in text.splitlines() if 'per position (kept / reached)' in x];persist()
  assert a['complete'] and a['exit_code']==0 and not a['warnings'] and not a['forced_kill'] and not floor
  assert not a['memory'].get('error');print('ARM_DONE',label,flush=True)
 for f,h in cfg['source_aux_hashes'].items():assert sha(f)==h,f
 for s in cfg['model_stats']:
  st=Path(s['path']).stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(s['device'],s['inode'],s['bytes'],s['mtime_ns'])
 assert sha(cfg['draft']['path'])==cfg['draft']['sha256'];idle();j['source_files_unchanged']=True;j['complete']=True;j['finished']=time.time();persist();print('COMPLETE',flush=True)
if __name__=='__main__':main()
