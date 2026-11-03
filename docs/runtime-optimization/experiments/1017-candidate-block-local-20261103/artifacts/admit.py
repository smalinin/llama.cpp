from pathlib import Path
import ctypes,ctypes.util,json,os,re,csv,subprocess,hashlib
P=Path(__file__).parent;cfg=json.load(open(P/'config.json'));assert json.load(open(P/'metadata-results.json'))['complete']
for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES']:os.environ[k]=cfg['environment'][k]
lib=ctypes.CDLL(ctypes.util.find_library('cudart')or'/usr/local/cuda/lib64/libcudart.so');n=ctypes.c_int();assert lib.cudaGetDeviceCount(ctypes.byref(n))==0
norm=lambda s:tuple(int(x,16)for x in re.split('[:.]',s.strip()))
nv={norm(r[2]):{'index':int(r[0]),'uuid':r[1].strip(),'bus':r[2].strip()}for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True).splitlines())}
mapping=[]
for i in range(n.value):
 b=ctypes.create_string_buffer(32);assert lib.cudaDeviceGetPCIBusId(b,32,i)==0;x=nv[norm(b.value.decode())];mapping.append({'logical':'CUDA'+str(i),'nvml_index':x['index'],'uuid':x['uuid'],'cuda_pci_bus':b.value.decode(),'nvml_pci_bus':x['bus']})
(P/'gpu-map.json').write_text(json.dumps({'environment':{k:os.environ[k]for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES']},'mapping':mapping},indent=2)+'\n')
refs=[(str(P/'metadata-100352-1.json'),str(P/'bench-results.json'),'on'),(str(P/'metadata-176128-0.json'),'/tmp/dsv41-candidate-pool-20261103/payload-176128/results.json',None)]
sources=[]
for meta,result,arm in refs:
 a=json.load(open(meta));b=json.load(open(result))
 if arm:b=next(x for x in b['arms']if x['label']==arm)
 assert b['exit_code']==0 and not b.get('forced_kill')
 sources.append((a,b['memory']['inference']))
out={'complete':False,'method':'Max observed per-device overhead from current candidate ON100352 clean benchmark and prior HC-sharing ON/poolON176128 payload (matching OFF budgets);additional64MiB guard;>=512MiB perGPU. Estimate,not actual memory gate.','reference_files':refs,'mapping':mapping,'candidates':[],'chosen_context':None}
out['reference_hashes']={p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()for a,b,_ in refs for p in[a,b]}
for ctx in [176128]:
 a=json.load(open(P/f'metadata-{ctx}-1.json'));rows=[]
 for i,x in enumerate(mapping):
  obs=[]
  for m,s in sources:
   assert m['memory'][i]['device']==a['memory'][i]['device']==x['logical'];z=s[x['uuid']];assert z['index']==x['nvml_index'];obs.append({'projected_free':m['memory'][i]['projected_free'],'observed_min_free':z['min_free'],'overhead':m['memory'][i]['projected_free']-z['min_free']})
  over=max(0,*(z['overhead']for z in obs));est=a['memory'][i]['projected_free']-over-64*1048576;rows.append({'device':x['logical'],'observations':obs,'projected_free':a['memory'][i]['projected_free'],'reserved_overhead':over,'extra_guard':64*1048576,'guarded_estimate_free':est})
 admitted=all(x['guarded_estimate_free']>=512*1048576 for x in rows);out['candidates'].append({'context':ctx,'admitted':admitted,'rows':rows})
 if admitted and out['chosen_context']is None:out['chosen_context']=ctx
assert out['chosen_context'];out['complete']=True;(P/'payload-admission.json').write_text(json.dumps(out,indent=2)+'\n');print('ADMIT',[(c['context'],c['admitted'],round(min(x['guarded_estimate_free']for x in c['rows'])/1048576,2))for c in out['candidates']],'CHOSEN',out['chosen_context'])
