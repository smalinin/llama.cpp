from pathlib import Path
import json,subprocess,shutil,os
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts';D=P/'shared-local';D.mkdir();assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cfg=json.loads((P/'launch-attribution/counts-results.json').read_text());env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(cfg['environment']);env.pop('LD_PRELOAD',None);env.pop('DSV_RESIDUAL_ARM',None);env.pop('DSV_RESIDUAL_LOG',None);env.pop('CUDA_SCALE_LAUNCH_QUEUES',None)
base=cfg['command'][:];candidate=base[:];i=candidate.index('--override-tensor')+1;candidate[i]+=r',blk\.6\.ffn_(gate|up)_shexp\.weight=CUDA1,blk\.29\.ffn_(gate|up)_shexp\.weight=CUDA4'
src=(P/'copies8/metadata.cpp').read_text();src=src.replace('j["routes"]=json::array();','j["shared_routes"]=json::array();j["routes"]=json::array();');src=src.replace('const auto&l=m->layers[il];','const auto&l=m->layers[il];if(il==6||il==29)for(auto*t:{l.ffn_gate_shexp,l.ffn_up_shexp,l.ffn_down_shexp})if(t){auto*b=t->view_src?t->view_src:t;j["shared_routes"].push_back({{"name",b->name},{"type",ggml_type_name(b->type)},{"bytes",ggml_nbytes(b)},{"buffer",ggml_backend_buffer_name(b->buffer)}});}')
(D/'metadata.cpp').write_text(src);cmd=[x.replace(str(P/'copies8/metadata.cpp'),str(D/'metadata.cpp')).replace(str(P/'copies8/metadata'),str(D/'metadata'))for x in json.loads((P/'copies8/build.json').read_text())['metadata_compile']]
with(D/'metadata-build.log').open('wb')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=R,check=True)
(D/'metadata-build.json').write_text(json.dumps({'command':cmd,'runtime_source_changes':False,'runtime_libraries':'Unchanged production, copies4, default launch queue.'},indent=2)+'\n')
meta=[]
for label,args in [('baseline',base),('candidate',candidate)]:
 with(D/(label+'-metadata.log')).open('wb')as f:subprocess.run([str(D/'metadata'),str(D/(label+'-metadata.json')),'measure',*args[1:]],stdout=f,stderr=subprocess.STDOUT,cwd=R,env=env,check=True)
 j=json.loads((D/(label+'-metadata.json')).read_text());assert j['complete']and j['gpu_complete_bundles']==15360 and j['routed_host_bytes']==0;meta.append(j);print(label,j['shared_routes'],[round(x['projected_free']/2**20,2)for x in j['memory']],flush=True)
a,b=meta
for k in ['routes','routed_gpu_bytes','routed_host_bytes','total_expert_bundles','gpu_complete_bundles','context','gpu_layers','split']:assert a[k]==b[k],k
assert len(a['shared_routes'])==len(b['shared_routes'])==6
for x,y in zip(a['shared_routes'],b['shared_routes']):
 assert (x['name'],x['type'],x['bytes'])==(y['name'],y['type'],y['bytes'])
 if 'gate_shexp'in y['name']or'up_shexp'in y['name']:assert y['buffer']==('CUDA1'if y['name'].startswith('blk.6.')else'CUDA4'),y
mapping=cfg['config']['environment']['CUDA_VISIBLE_DEVICES'].split(',');gates=[]
for i,(x,y)in enumerate(zip(a['memory'],b['memory'])):
 actual=min(cfg['memory'][ph][mapping[i]]['min_free']for ph in ['load','inference']);pred=y['projected_free']-max(0,x['projected_free']-actual)-64*2**20;gates.append({'device':x['device'],'conservative_free_mib':pred/2**20,'admit':pred>=512*2**20})
(D/'admission.json').write_text(json.dumps({'complete':True,'admit':all(x['admit']for x in gates),'devices':gates,'note':'Same context100352; production copies4/queue unset. No_alloc plus actual baseline overhead plus64MiB guard. Actual run still required.','command':candidate},indent=2)+'\n');assert all(x['admit']for x in gates)
# Fresh config-only screen; preserve same mature probe, default queue and copy4.
shutil.copy2(P/'memory_monitor.py',D/'memory_monitor.py');(D/'probe.so').symlink_to('../launch-attribution/probe.so');s=(P/'launch-attribution/run.py').read_text().replace('/artifacts/launch-attribution','/artifacts/shared-local').replace('19095','19098').replace("='prefill-launch-attribution'","='prefill-shared-local'")
assert "cmd=cfg['command'][:]"in s
s=s.replace("cmd=cfg['command'][:];", "cmd=cfg['command'][:];cmd[cmd.index('--override-tensor')+1]+=r',blk\\.6\\.ffn_(gate|up)_shexp\\.weight=CUDA1,blk\\.29\\.ffn_(gate|up)_shexp\\.weight=CUDA4';")
s=s.replace("'kind':'", "'kind':'Placement-only shared gate/up layers6->CUDA1 and29->CUDA4; production copy4/queue unset. ")
(D/'run.py').write_text(s);(P.parent/'.gitignore').open('a').write('\nartifacts/shared-local/metadata\n')
shutil.copy2('/tmp/prepare-residual-shared-local.py',P/'prepare-residual-shared-local.py')
print('ADMITTED: use shared-local/run.py once',flush=True)
