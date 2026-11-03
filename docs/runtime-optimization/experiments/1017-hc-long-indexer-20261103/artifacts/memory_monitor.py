import ctypes,ctypes.util,threading,time
from pathlib import Path
class MemoryMonitor:
 def __init__(self,pid):
  self.pid=pid;self.lib=ctypes.CDLL(ctypes.util.find_library('nvidia-ml')or'libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0;count=ctypes.c_uint();assert self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count))==0;self.devices=[]
  for i in range(count.value):
   h=ctypes.c_void_p();assert self.lib.nvmlDeviceGetHandleByIndex_v2(i,ctypes.byref(h))==0;u=ctypes.create_string_buffer(96);assert self.lib.nvmlDeviceGetUUID(h,u,96)==0;self.devices.append((i,h,u.value.decode()))
  self.lock=threading.Lock();self.stop_event=threading.Event();self.phase='load';self.stats={'load':{},'inference':{}};self.samples=0;self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def sample(self):
  class Mem(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
  gpu=[]
  for i,h,u in self.devices:
   m=Mem();assert self.lib.nvmlDeviceGetMemoryInfo(h,ctypes.byref(m))==0;gpu.append({'index':i,'uuid':u,'total':m.total,'used':m.used,'free':m.free})
  rss=0
  try:
   for line in Path(f'/proc/{self.pid}/status').read_text().splitlines():
    if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
  except FileNotFoundError:pass
  with self.lock:
   s=self.stats[self.phase];s['samples']=s.get('samples',0)+1;s['max_rss']=max(s.get('max_rss',0),rss);s['last_rss']=rss
   for g in gpu:
    k=g['uuid'];x=s.setdefault(k,dict(index=g['index'],total=g['total'],peak_used=0,min_free=g['total']));x['peak_used']=max(x['peak_used'],g['used']);x['min_free']=min(x['min_free'],g['free']);x['last_used']=g['used'];x['last_free']=g['free']
  return gpu
 def loop(self):
  while not self.stop_event.wait(.1):
   try:self.sample()
   except Exception as e:
    with self.lock:self.stats['error']=repr(e)
    break
 def inference(self):
  with self.lock:self.phase='inference'
  self.sample()
 def finish(self):
  self.sample();self.stop_event.set();self.thread.join()
  with self.lock:return self.stats
