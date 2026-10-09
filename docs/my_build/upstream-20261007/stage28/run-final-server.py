#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
spec=importlib.util.spec_from_file_location('parallel_stage2',R.parent/'stage2/run-parallel.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
GPU=json.loads((R.parent/'stage9/current-server/off/manifest.json').read_text())['gpu_order']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(label,old,unified,kind,cache):
 lib=R.parent/'stage27/candidate-bin' if old else R/'final-bin'
 hashes=json.loads((R.parent/'stage27/build-manifest.json' if old else R/'final-build-manifest.json').read_text())['candidate_hashes']
 assert all(sha(lib/n)==h for n,h in hashes.items())
 code=base.source.replace("'-fit', 'on', '--fit-target', '3072',", "'-fit', 'off', '-ngl', '99', '--tensor-split', '1,1,1,1,1,0.4',")
 code=code.replace("'--metrics',", "'--metrics', "+repr('--kv-unified' if unified else '--no-kv-unified')+',')
 code=code.replace("env = os.environ.copy()", "env = os.environ.copy()\n    for key in list(env):\n        if key.startswith('LLAMA_GLM5_') or key in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','NVIDIA_TF32_OVERRIDE'): env.pop(key)\n    env['LLAMA_GLM5_POOL_CACHE'] = "+repr(str(cache)))
 code=code.replace("            save(directory / 'props.json'", "            print('READY', round(result['startup_s'],1), flush=True)\n            save(directory / 'props.json'")
 namespace={'__name__':'stage28_parallel'};exec(compile(code,str(R.parent/'stage2/run-parallel.py'),'exec'),namespace);namespace['GPU_ORDER']=GPU
 profile=namespace['PROFILES']['glm5next'];profile['spec_type']=kind
 if kind=='draft-dflash':profile['draft']=Path('/home/sergei/.models/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/GLM-5.3-Flash-DFlash2-Q8_0.gguf')
 out=R/'server-runs'/label;out.mkdir(parents=True,exist_ok=False)
 inputs=[profile['model']]
 inputs=list(profile['model'].parent.glob('GLM-5.3-Flash-UD-Q4_K_XL-*-of-*.gguf'))+([profile['draft']] if 'draft' in profile else [])
 extra={'label':label,'library_hashes':hashes,'pool_cache':cache,'kv_unified':unified,'spec_type':kind,'models':{str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in inputs},'source_hashes':json.loads((R/'final-build-manifest.json').read_text())['sources']}
 (out/'control-manifest.json').write_text(json.dumps(extra,indent=2)+'\n');print('CONTROL',label,flush=True)
 result=namespace['run_case']('glm5next',True,out,lib/'llama-server');result['label']=label
 for p,v in extra['models'].items():assert Path(p).stat().st_size==v['size'] and Path(p).stat().st_mtime_ns==v['mtime_ns']
 return result
if __name__=='__main__':
 cases={'mtp-final-unified':(False,True,'draft-mtp',1)}
 p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',choices=cases,default=list(cases));args=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 for label in args.cases:
  run(label,*cases[label])
