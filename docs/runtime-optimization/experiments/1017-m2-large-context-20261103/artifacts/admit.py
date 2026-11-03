from pathlib import Path
import ctypes,ctypes.util,json,os,re,csv,subprocess,hashlib
P=Path(__file__).parent;cfg=json.load(open(P/'config.json'));m=json.load(open(P/'metadata-results.json'));assert m['complete']
for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES']:os.environ[k]=cfg['environment'][k]
lib=ctypes.CDLL(ctypes.util.find_library('cudart')or'/usr/local/cuda/lib64/libcudart.so');n=ctypes.c_int();assert lib.cudaGetDeviceCount(ctypes.byref(n))==0
norm=lambda s:tuple(int(x,16)for x in re.split('[:.]',s.strip()))
nv={norm(r[2]):{'index':int(r[0]),'uuid':r[1].strip(),'bus':r[2].strip()}for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True).splitlines())};mapping=[]
for i in range(n.value):
 b=ctypes.create_string_buffer(32);assert lib.cudaDeviceGetPCIBusId(b,32,i)==0;raw=b.value.decode();x=nv[norm(raw)];mapping.append({'logical':'CUDA'+str(i),'nvml_index':x['index'],'uuid':x['uuid'],'runtime_pci_bus':raw,'nvml_pci_bus':x['bus']})
(P/'gpu-map.json').write_text(json.dumps({'environment':{k:os.environ[k]for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES']},'mapping':mapping},indent=2)+'\n')
native=json.load(open(cfg['reference_native_results']));fresh=next(v for v in native['variants']if v['variant']=='native');assert fresh['complete']and fresh['exit_code']==0
source_pairs=[(P/'metadata-100352-default.json',None,fresh['memory']['inference']), (Path('/tmp/dsv41-feature-local-long-20261103/metadata-176128.json'),Path('/tmp/dsv41-feature-local-long-20261103/payload-176128/results.json'),None),(Path('/tmp/dsv41-prefix-mask-fit-20261102/pinned-176128-default.json'),Path('/tmp/dsv41-prefix-mask-fit-20261102/payload-pinned-176128-default/results.json'),None)]
sources=[]
for meta,result,obs in source_pairs:
 a=json.load(open(meta));b=json.load(open(result))if result else None
 if b:assert b['complete']and b['exit_code']==0
 sources.append((a,obs or b['memory']['inference']))
out={'complete':False,'method':'max per-device measured overhead across fresh M2 ctx100352, historical LOCAL176128, historical CPU176128; additional64MiB guard; require >=512MiB. Estimate only.','source_files':[str(p)for a,b,_ in source_pairs for p in[a,b]if p]+[cfg['reference_native_results']],'mapping':mapping,'candidates':[],'chosen_context':None}
out['source_hashes']={p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()for p in out['source_files']}
for ctx in [176128,163840,159744]:
 c=json.load(open(P/f'metadata-{ctx}-default.json'));rows=[]
 for i,x in enumerate(mapping):
  obs=[]
  for a,b in sources:
   assert a['memory'][i]['device']==c['memory'][i]['device']==x['logical'];v=b[x['uuid']];assert v['index']==x['nvml_index'];obs.append({'projected_free':a['memory'][i]['projected_free'],'observed_min_free':v['min_free'],'overhead':a['memory'][i]['projected_free']-v['min_free']})
  over=max(0,*(v['overhead']for v in obs));est=c['memory'][i]['projected_free']-over-64*1048576;rows.append({'device':x['logical'],'observations':obs,'projected_free':c['memory'][i]['projected_free'],'reserved_overhead':over,'extra_guard':64*1048576,'guarded_estimate_free':est})
 admitted=all(r['guarded_estimate_free']>=512*1048576 for r in rows);out['candidates'].append({'context':ctx,'admitted':admitted,'rows':rows})
 if admitted and out['chosen_context']is None:out['chosen_context']=ctx
assert out['chosen_context'];out['complete']=True;(P/'payload-admission.json').write_text(json.dumps(out,indent=2)+'\n');print('CHOSEN',out['chosen_context']);print([(c['context'],c['admitted'],round(min(r['guarded_estimate_free']for r in c['rows'])/1048576,2))for c in out['candidates']])
