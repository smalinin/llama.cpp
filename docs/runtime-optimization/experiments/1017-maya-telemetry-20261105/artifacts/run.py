from pathlib import Path
from urllib.request import Request,urlopen
import os,json,time,subprocess,signal,hashlib,shutil,threading,queue,socket,traceback,re,sys
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-maya-telemetry-20261105/artifacts';BASE=R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts';NSYS='/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'
sys.path.insert(0,str(P))
from memory_monitor import MemoryMonitor

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):
 q=p.with_suffix(p.suffix+'.new');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def stats(pid):
 a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();io={k:int(v)for k,v in [l.split(':')for l in Path(f'/proc/{pid}/io').read_text().splitlines()]};return {'cpu_s':(int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK'),'minor_faults':int(a[7]),'major_faults':int(a[9]),'io':io,'mono_ns':time.monotonic_ns()}
def idle():assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip()
def main():
 cfg=json.loads((P/'config.json').read_text());ids=json.loads((P/'inputs.json').read_text())['tokens'];native=json.loads((BASE/'results.json').read_text());assert native['complete'];assert not(P/'results.json').exists()
 base={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV','DS4_','STRATA_','MAYA_'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};base.update({k:cfg['environment'][k]for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES']})
 j={'complete':False,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'native_baseline':str(BASE/'results.json'),'config':cfg,'nsys_version':subprocess.check_output([NSYS,'--version'],text=True).strip(),'cpu_perf_sampling':'unavailable: kernel perf_event_paranoid=4, nsys environment check failed; no system settings changed','probe_hashes':{f.name:sha(f)for f in P.glob('*.so')},'scope':'Diagnostic only, not native speed. Same new-host source/binaries/UUIDs, sidecar-onK5/pmin.5/cum0/full. Two CUDA/NVTX/OSRT captures/engine,8K and98K,128greedy outputs, warm/pre/profile/post controls. Generic Unix socket watcher controls capture/PP->TG markers, drains after response. Sampled launch/memcpy/sync callstacks, ours-only decode/ubatch/proposal/bridge NVTX wrappers, no eval callbacks or source changes. CPU/I/O counters and memory floor512. Maya CPU_STATS1, DS4_PROF intentionally absent to avoid profiler mutex/event serialization of flow. Maya same-cohort lane timings copied with pack-path metadata remap, initial expert frequencies unchanged. Phase split is client first-output observation, not exact device fence. Native timings remain in separate report.','arms':[]}
 def persist():save(P/'results.json',j)
 persist();capture=0
 for engine in ['maya','ours']:
  idle()
  for f,h in cfg['source_library_hashes'].items():assert sha(f)==h,f
  label=engine;d=P/engine;d.mkdir(exist_ok=False);e=base.copy();sockpath=f'/tmp/d5tm-{os.getuid()}-{engine}.sock';e.update(DSV_PROFILE_SOCKET=sockpath,DSV_PROFILE_LOG=str(d/'probe.jsonl'),DSV_PROFILE_PID=str(d/'pid'))
  if engine=='ours':
   e.update(cfg['environment']);cmd=cfg['command'][:];cmd[cmd.index('--port')+1]='19091';cmd[cmd.index('--alias')+1]='telemetry-ours';preload=str(P/'control.so')+':'+str(P/'roles.so')
  else:
   original=next(a for a in native['arms']if a['label']=='maya-on');cmd=original['command'][:];cmd[cmd.index('--pack')+1]=str(d/'pack');shutil.copytree(BASE/'pack-base',d/'pack')
   oldpack=BASE/'maya-on/pack';lane=(oldpack/'lane-times.txt').read_text();newlane=lane.replace(str(oldpack),str(d/'pack'));(d/'pack/lane-times.txt').write_text(newlane);save(d/'lane-remap.json',{'source':str(oldpack/'lane-times.txt'),'source_sha256':sha(oldpack/'lane-times.txt'),'new_sha256':sha(d/'pack/lane-times.txt'),'operation':'only pack-path text changed; same new-host measured timings and same CPU threads, GPU UUID/PCI'});e.update(original['environment']);e['DS4_CPU_STATS']='1';preload=str(P/'control.so')
  launch=[NSYS,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--capture-range=cudaProfilerApi','--capture-range-end=repeat:2','--cuda-graph-trace=node','--osrt-threshold=100000','--force-overwrite=false','--output='+str(d/'capture'),'env','LD_PRELOAD='+preload,*cmd]
  a={'label':label,'complete':False,'command':cmd,'launch':launch,'environment':{k:v for k,v in e.items()if k.startswith(('LLAMA_','GGML_','CUDA_','DSV','STRATA_','DS4_','LD_'))},'requests':[]};j['arms'].append(a);persist();proc=None;pid=None;mon=None;reader=None;guard=None;control=None;ctrlsock=None;halt=threading.Event();floor=[]
  try:
   with(d/'server.log').open('wb')as log:
    proc=subprocess.Popen(launch,cwd=d,env=e,stdin=subprocess.PIPE if engine=='maya'else subprocess.DEVNULL,stdout=subprocess.PIPE if engine=='maya'else log,stderr=log if engine=='maya'else subprocess.STDOUT,text=engine=='maya',bufsize=1 if engine=='maya'else -1,start_new_session=True);save(P/'active.json',{'wrapper':proc.pid,'label':label})
    started=time.monotonic()
    while not(d/'pid').exists():assert proc.poll()is None;assert time.monotonic()-started<120;time.sleep(.1)
    pid=int((d/'pid').read_text());a['pid']=pid;mon=MemoryMonitor(pid)
    def watch():
     while not halt.wait(.2):
      with mon.lock:bad=any(v.get('min_free',512*1048576)<512*1048576 for ph in ['load','inference']for v in mon.stats[ph].values()if isinstance(v,dict))
      if bad:
       floor.append(time.monotonic())
       try:os.kill(pid,signal.SIGTERM)
       except ProcessLookupError:pass
       return
    guard=threading.Thread(target=watch,daemon=True);guard.start()
    ctrlsock=socket.socket(socket.AF_UNIX);ctrlsock.settimeout(120);ctrlsock.connect(sockpath);control=ctrlsock.makefile('rw')
    def send(message):
     control.write(message+'\n');control.flush();v=control.readline();assert v.startswith('ACK'),v;return v.strip()
    if engine=='ours':
     def api(body=None):
      with urlopen(Request('http://127.0.0.1:19091'+('/health'if body is None else'/completion'),data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=1500 if body else 3)as f:return json.load(f)
     while True:
      assert proc.poll()is None
      try:
       if api().get('status')=='ok':break
      except OSError:pass
      assert time.monotonic()-started<1500;time.sleep(1)
     def request(tokens,count,onfirst=None):
      body=dict(prompt=tokens,n_predict=count,temperature=0,top_k=1,top_p=1,min_p=0,repeat_penalty=1,seed=1234,ignore_eos=True,cache_prompt=False,return_tokens=True,id_slot=0,stream=bool(onfirst))
      if onfirst is None:return api(body)
      out=[];events=[];first=True;last=None
      with urlopen(Request('http://127.0.0.1:19091/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=1500)as f:
       for line in f:
        if not line.startswith(b'data: '):continue
        raw=line[6:].strip()
        if raw==b'[DONE]':break
        v=json.loads(raw);events.append(v)
        if v.get('tokens'):
         if first:onfirst();first=False
         out+=v['tokens']
        if v.get('stop'):last=v
      assert last is not None and not first
      return {'tokens':out,'timings':last['timings'],'final':last,'events':events}
    else:
     q=queue.Queue();transcript=(d/'protocol.log').open('w')
     def read():
      for l in proc.stdout:transcript.write(l);transcript.flush();q.put(l.strip())
      q.put(None)
     reader=threading.Thread(target=read,daemon=True);reader.start()
     def get():
      l=q.get(timeout=1500)
      if l is None or l.startswith('ERR '):raise RuntimeError(str(l))
      return l
     while not get().startswith('READY '):pass
     def request(tokens,count,onfirst=None):
      proc.stdin.write(f'GEN {count} temperature=0 top_p=1 top_k=1 min_p=0 seed=1234 dspark_conf=0.5 dspark_cum=0 '+','.join(map(str,tokens))+'\n');proc.stdin.flush();out=[];events=[]
      while True:
       l=get()
       if l.startswith('T '):
        if not out and onfirst:onfirst()
        out.append(int(l.split()[1]))
       elif l.startswith('STAT '):events.append(l)
       elif l.startswith('DONE '):return {'tokens':out,'done':l,'stats':events}
    mon.inference();a['ready_seconds']=time.monotonic()-started;persist();print('READY',engine,flush=True)
    for n in [8192,98304]:
     plan=['warmup','warmup','pre','profile','post']if n==8192 else['warmup','pre','profile','post']
     for kind in plan:
      request([97],1);traced=kind=='profile';capture+=int(traced);before=None;first=None;first_wall=None
      if traced:start_ack=send(f'START {capture}')
      before=stats(pid);ts=time.monotonic()
      def first_output():
       nonlocal first,first_wall
       first=stats(pid);first_wall=time.monotonic()-ts
       if traced:send('TG')
      v=request(ids[:n],128,first_output if traced else None);wall=time.monotonic()-ts;after=stats(pid)
      if traced:stop_ack=send('STOP')
      assert len(v['tokens'])==128
      if engine=='ours':t=v['timings'];assert t['cache_n']==0 and t['prompt_n']==n and t['predicted_n']==128
      else:
       f=v['done'].split();assert int(f[1])==128 and int(f[2])==n and int(f[8])==0;t={'prompt_n':n,'prompt_ms':float(f[3]),'predicted_ms':float(f[4]),'draft_n_accepted':int(f[6]),'draft_n':int(f[7])}
      row={'n':n,'kind':kind,'capture_id':capture if traced else None,'response':v,'timings':t,'tokens_sha256':hashlib.sha256(json.dumps(v['tokens'],separators=(',',':')).encode()).hexdigest(),'wall_s':wall,'before':before,'first':first,'after':after,'first_wall_s':first_wall}
      if traced:row['control_ack']=[start_ack,stop_ack]
      a['requests'].append(row);persist();assert not floor;print('CASE',engine,n,kind,round(n*1000/t['prompt_ms'],1),round(127000/t['predicted_ms'],1),'cpu',round(after['cpu_s']-before['cpu_s'],2),flush=True)
    a['complete']=True
  except Exception as ex:a['error']=repr(ex);a['traceback']=traceback.format_exc();raise
  finally:
   if control:control.close()
   if ctrlsock:ctrlsock.close()
   if pid:
    try:os.kill(pid,signal.SIGTERM)if engine=='ours'else(proc.stdin.write('QUIT\n'),proc.stdin.flush())
    except (ProcessLookupError,BrokenPipeError):pass
   if proc:
    try:proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
     if pid:
      try:os.kill(pid,signal.SIGKILL)
      except ProcessLookupError:pass
     proc.kill();proc.wait();a['forced_kill']=True
   halt.set()
   if guard:guard.join(timeout=3)
   if reader:reader.join(timeout=3);transcript.close()
   if mon:a['memory']=mon.finish()
   a['exit_code']=None if proc is None else proc.returncode;a['floor_breach']=floor
   text=(d/'server.log').read_text(errors='replace');a['warnings']=[l for l in text.splitlines()if re.search('does not match expectation|out of memory|failed to allocate|CUDA error',l,re.I)];a['nsys_diagnostics']=[l for l in text.splitlines()if re.search('error|warning|incomplete|dropped',l,re.I)];a['layout']=[l for l in text.splitlines()if re.search(r'CUDA\d+ layers|routed experts on the CPU lane|model buffer size',l)];persist()
   try:Path(sockpath).unlink()
   except FileNotFoundError:pass
  assert a['complete'] and a['exit_code']==0 and not a.get('forced_kill')
  assert not floor and not a['warnings']
  a['traces']=[]
  for f in sorted(d.glob('*.nsys-rep')):
   sql=f.with_suffix('.sqlite')
   with(d/(f.stem+'-export.log')).open('wb')as out:subprocess.run([NSYS,'export','--type=sqlite','--force-overwrite=false','--output='+str(sql),str(f)],stdout=out,stderr=subprocess.STDOUT,check=True)
   a['traces'].append({'report':str(f),'sqlite':str(sql),'report_sha256':sha(f),'sqlite_sha256':sha(sql)})
  assert len(a['traces'])==2;persist();print('ARM_DONE',engine,flush=True)
 idle();j['complete']=True;persist();print('TELEMETRY_COMPLETE',flush=True)
if __name__=='__main__':main()
