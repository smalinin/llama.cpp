from pathlib import Path
import json,os,subprocess,re,hashlib,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-kv-owner-audit-20261105/artifacts';Old=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts';j=json.loads((Old/'launch-attribution/counts-results.json').read_text());assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env={k:v for k,v in os.environ.items()if not k.startswith(('LLAMA_','GGML_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(j['environment']);env.update(LD_PRELOAD=str(P/'probe.so'),DSV_OWNER_SMOKE_JSON=str(P/'scheduler-owner-smoke.json'));env.pop('DSV_RESIDUAL_ARM',None);env.pop('DSV_RESIDUAL_LOG',None)
with(P/'scheduler.log').open('wb')as f:r=subprocess.run([str(P/'scheduler-harness'),str(P/'scheduler-results.json')],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=R)
assert r.returncode==0,r.returncode;u=json.loads((P/'scheduler-results.json').read_text());assert u['complete']and len(u['cases'])==42;print('42 scheduler forwarding cases plus real graph owner smoke passed.',flush=True)
s=(P/'run.py').read_text();s=re.sub(r"'kind':'[^']*'", "'kind':'External read-only KV owner audit: host API scopes with graph nodes, src/view roots, layout/flags and public buffer owners on selected target ubatches. No CUDA API interposition, added GPU events, eval callbacks, device reads or math/source/default changes. Same-process unarmed pre/post controls, default CPU lookup ON.'",s,count=1)
needle="cfg['source_files']['ggml/src/ggml-backend.cpp']=sha(R/'ggml/src/ggml-backend.cpp')";assert needle in s
s=s.replace(needle,needle+"\n for f in ['src/models/deepseek41.cpp','src/llama-kv-cache.cpp','src/llama-kv-cache-dsv4.cpp','ggml/src/ggml-alloc.c','ggml/src/ggml-cuda/set-rows.cu']:cfg['source_files'][f]=sha(R/f)")
(P/'run.py').write_text(s);shutil.copy2('/tmp/run-kv-owner-audit.py',P/'run-controller.py')
with(P/'run.log').open('wb')as f:r=subprocess.run(['python3',str(P/'run.py')],stdout=f,stderr=subprocess.STDOUT,cwd=R)
assert r.returncode==0,r.returncode;print('OWNER_AUDIT_INFERENCE_COMPLETE',flush=True)
