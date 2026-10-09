from pathlib import Path
import argparse,sys,json,os,socket,subprocess,time,urllib.error,hashlib,struct
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));import run_baseline as base
p=argparse.ArgumentParser();p.add_argument('profile');p.add_argument('label');p.add_argument('--baseline',action='store_true');p.add_argument('--native',action='store_true');p.add_argument('--faults',action='store_true');p.add_argument('--restart',action='store_true');p.add_argument('--q8',action='store_true');a=p.parse_args()
out=R/a.label;out.mkdir(exist_ok=False);slots=out/'slots';slots.mkdir();lib=(R.parent/'stage33/final-bin' if a.baseline else R/'candidate-bin').resolve()
profile=base.PROFILES[a.profile] if a.profile!='tiny' else {'model':Path('/home/sergei/Github/llama.cpp/build-glm53/tinyllamas/stories15M-q4_0.gguf')}
if a.profile=='glm-dsa':profile=dict(profile,model=Path('/home/sergei/.models/antirez/GLM-5.3-GGUF/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf'))
np=1 if a.profile in ['deepseek41','glm-dsa'] else 2
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
cmd=[str(lib/'llama-server'),'-m',str(profile['model']),'-c',str(8192*np),'-b','2048','-ub','512','-np',str(np),'-ctk','q8_0' if a.q8 else 'f16','-ctv','q8_0' if a.q8 else 'f16','-t','12','-tb','12','-fit','off','-ngl','99','--tensor-split','1,1,1,1,1,0.4','--split-mode','layer','--load-mode','none','--lazy-mode','auto','--host','127.0.0.1','--port',str(port),'--slot-save-path',str(slots)+'/', '--cache-ram','0','--jinja']
if not a.native:
 cmd+=['--spec-type',profile.get('spec_type','draft-mtp'),'--spec-draft-n-max','3','--spec-draft-p-min','0','--spec-draft-ngl','99']
 if 'draft' in profile:cmd+=['--model-draft',str(profile['draft'])]
env=os.environ.copy()
for k in list(env):
 if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_','LLAMA_DSV41_','GGML_CUDA_DISABLE_GRAPHS')):env.pop(k)
env.update(LD_LIBRARY_PATH=str(lib),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='GPU-3972c9ef-dd88-6b95-ac0c-fc9b0eb86f76,GPU-9e5fb796-93b1-b72d-1fd8-438d93aac573,GPU-8e5bef34-d86f-b6ed-3419-3d0c1be7dd11,GPU-7f297c35-ddfe-c2dd-9d94-fbf1477b86b6,GPU-dd8b5c70-2afd-3c20-7b36-59469a2b411a,GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783')
def sha(f):
 with Path(f).open('rb') as g:return hashlib.file_digest(g,'sha256').hexdigest()
manifest={'cmd':cmd,'env':{k:env[k] for k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']},'hashes':{f.name:sha(f) for f in lib.iterdir() if f.is_file()}}
base.save(out/'manifest.json',manifest);result={};proc=None;log=None

def call(label,path,data=None,status=200):
 if data is not None:base.save(out/(label+'-request.json'),data)
 try:res=base.api(port,path,data,timeout=900);code=200
 except urllib.error.HTTPError as e:code=e.code;res=json.load(e)
 base.save(out/(label+'-response.json'),res);print(label,code,res.get('timings',res if code!=200 else {}),flush=True);assert code==status,res;return res

def start(label):
 global proc,log
 log=(out/(label+'.log')).open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+900
 while True:
  assert proc.poll() is None,'server died'
  try:
   if base.api(port,'/health',timeout=2).get('status')=='ok':break
  except (urllib.error.URLError,TimeoutError):pass
  assert time.monotonic()<deadline;time.sleep(1)
 maps=Path(f'/proc/{proc.pid}/maps').read_text();libs=sorted({x.split()[-1] for x in maps.splitlines() if '/libggml' in x or '/libllama' in x});assert all(Path(x).parent==lib for x in libs),libs
 manifest[label+'-loaded']=libs;base.save(out/'manifest.json',manifest)

def stop():
 global proc,log
 if proc and proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 if log:log.close()

def complete(label,tokens,slot=0,n=32):
 return call(label,'/completion',{'prompt':tokens,'id_slot':slot,'n_predict':n,'temperature':0,'seed':1234,'cache_prompt':True,'return_tokens':True,'stream':False,'ignore_eos':True})
def slot(label,act,idx=0,filename='state.bin',status=200):return call(label,f'/slots/{idx}?action={act}',{} if act=='erase' else {'filename':filename},status)
try:
 start('server')
 tokens=call('tokenize','/tokenize',{'content':'Read these facts.\n'+''.join(f'Entry {i}: the value is {1000+i}; the label is blue.\n' for i in range(32))+'Explain the pattern and list ten values.\nAnswer:', 'add_special':True,'parse_special':True})['tokens']
 if a.profile=='tiny':tokens=tokens[:40]
 if a.profile=='deepseek41':tokens=json.loads((R.parent/'stage33/final-off/ratio2-long-fresh-request.json').read_text())['prompt']
 initial=complete('initial',tokens,slot=np-1,n=32);prefix=tokens+initial['tokens'];saved=slot('save','save',np-1);expected=saved['n_saved']
 control=complete('control',prefix,slot=np-1)
 slot('erase','erase',np-1);rest=slot('restore','restore',0)
 restored=complete('restored',prefix)
 result.update(expected_cached=expected,control_cache=control['timings']['cache_n'],restored_cache=restored['timings']['cache_n'],tokens_equal=control['tokens']==restored['tokens'],text_equal=control['content']==restored['content'],bytes_equal=saved['n_written']==rest['n_read'],draft_n=restored['timings'].get('draft_n'))
 base.save(out/'result.json',result)
 if not a.baseline:assert result['tokens_equal'] and result['text_equal'] and result['bytes_equal'] and result['control_cache']==expected and result['restored_cache']==expected,result
 if a.restart:
  stop();start('restarted');slot('restart-restore','restore');again=complete('restart-continuation',prefix);result['restart_equal']=again['tokens']==control['tokens'];result['restart_cache']=again['timings']['cache_n'];assert result['restart_equal'] and result['restart_cache']==expected
 if a.faults:
  raw=(slots/'state.bin').read_bytes();marker=struct.pack('<QQ',0x46524453,1);offset=raw.rfind(marker);assert offset>=0
  header=list(struct.unpack_from('<5Q',raw,offset));result['trailer']={'offset':offset,'header':header}
  faults=[]
  for name,cut,pos,val in [('header-cut',offset+10,None,None),('payload-cut',len(raw)-1,None,None),('bad-magic',None,offset,0),('bad-version',None,offset+8,2),('bad-size',None,offset+24,2**64-1),('bad-signature',None,None,None),('extra',None,None,None),('bad-draft',None,None,None),('bad-spec',None,None,None)]:
   if name=='bad-spec' and not header[4]:continue
   data=bytearray(raw[:cut]) if cut is not None else bytearray(raw)
   if pos is not None:struct.pack_into('<Q',data,pos,val)
   if name=='bad-signature':data[offset+40]^=1
   if name=='extra':data+=b'junk'
   if name=='bad-draft':data[offset+40+header[2]]=255
   if name=='bad-spec':data[-header[4]:]=b'\xff'*header[4]
   (slots/'fault.bin').write_bytes(data)
   slot(name+'-restore','restore',filename='fault.bin',status=400)
   check=complete(name+'-usable',tokens[:30],n=4);assert check['timings']['cache_n']==0
   faults.append(name)
  result['faults_rejected']=faults
  (slots/'legacy.bin').write_bytes(raw[:offset]);slot('legacy-restore','restore',filename='legacy.bin');legacy=complete('legacy-continuation',prefix);result['legacy_cache']=legacy['timings']['cache_n'];result['legacy_tokens_equal']=legacy['tokens']==control['tokens'];assert result['legacy_cache']==0
 base.save(out/'result.json',result);print('RESULT',result,flush=True)
finally:stop()
