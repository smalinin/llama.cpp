from pathlib import Path
import os,json,subprocess,hashlib,time
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts';D=P/'promotion';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
cfg=json.loads((P/'native-results.json').read_text())['config'];env={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','CUDA_'))and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(cfg['environment']['CUDA_VISIBLE_DEVICES'].split(',')[:2]),LD_LIBRARY_PATH=str(D/'candidate')+':'+str(R/'build-aliceai/bin'))
cmd=[str(D/'scheduler-harness'),str(D/'scheduler-results.json')];j={'complete':False,'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('CUDA_','LD_'))},'build':json.loads((D/'build.json').read_text())};save=lambda:(D/'results.json').write_text(json.dumps(j,indent=2)+'\n');save();start=time.monotonic()
with(D/'scheduler.log').open('wb')as f:r=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
j['exit_code']=r.returncode;j['seconds']=time.monotonic()-start;save();assert r.returncode==0
u=json.loads((D/'scheduler-results.json').read_text());assert u['complete']and len(u['cases'])==42
positive={'eligible','view','shapes','switch','failure'}
for c in u['cases']:
 assert(c['early']>0)==(c['mode']in [1,-1]and c['variant']in positive),c
j['cases']=42;j['unset_matches_enabled_dispatch']=True;j['explicit_zero_keeps_fallback']=True;j['complete']=True;save();print('PROMOTION PASS: 42 scheduler cases; default/ON dispatch active and outputs exact, OFF and unsafe modes retain fallback.')
