from pathlib import Path
import json,os,sys,socket,subprocess,time,urllib.error
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));import run_baseline as base
source=R/'pytest-tmp/test_slot_save_restore_draft0/draft.bin';assert source.exists()
out=R/'native-compat';out.mkdir(exist_ok=True);lib=R/'candidate-bin'
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
cmd=[str(lib/'llama-server'),'-m','/home/sergei/Github/llama.cpp/build-glm53/tinyllamas/stories15M-q4_0.gguf','-c','4096','-np','2','-ngl','0','-fa','off','--host','127.0.0.1','--port',str(port),'--slot-save-path',str(source.parent)+'/', '--cache-ram','0']
env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(lib),CUDA_VISIBLE_DEVICES='');base.save(out/'manifest.json',{'cmd':cmd,'file_bytes':source.stat().st_size})
with (out/'server.log').open('w') as log:
 proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for i in range(120):
   assert proc.poll() is None
   try:
    if base.api(port,'/health',timeout=2).get('status')=='ok':break
   except urllib.error.URLError:time.sleep(.1)
  res=base.api(port,'/slots/0?action=restore',{'filename':'draft.bin'});base.save(out/'result.json',res);assert res['n_read']==source.stat().st_size;print(res)
 finally:proc.terminate();proc.wait(timeout=30)
