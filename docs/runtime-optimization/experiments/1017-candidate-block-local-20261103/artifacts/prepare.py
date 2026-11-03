from pathlib import Path
import json,shutil,subprocess,hashlib
P=Path(__file__).parent;O=Path('/tmp/dsv41-candidate-pool-20261103');R=Path('/home/sergei/Github/llama.cpp')
for n in ['memory_monitor.py','inputs.json','metadata-trace.cpp']:shutil.copyfile(O/n,P/n)
cfg=json.load(open(O/'config.json'));cfg['source_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();cfg['runtime_patch']=subprocess.check_output(['git','diff','--','src'],cwd=R,text=True);cfg['environment']['LLAMA_DSV41_CANDIDATE_POOL_LOCAL']='1';cfg['source_library_hashes']={n:hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()for n in cfg['source_library_hashes']};(P/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
shutil.copyfile('/tmp/dsv41-hc-on-long-indexer-20261103/metadata.cpp',P/'metadata.cpp')
base=['g++','-std=c++17','-O2','-Iinclude','-Icommon','-Isrc','-Iggml/include','-Ivendor','-Lbuild-aliceai/bin','-Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin']
with(P/'compile-helpers.log').open('wb')as f:
 subprocess.run(base+[str(P/'metadata.cpp'),'-lllama-common','-lllama','-lggml','-lggml-base','-ldl','-lpthread','-o',str(P/'metadata')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(base+[str(P/'metadata-trace.cpp'),'-fPIC','-shared','-ldl','-o',str(P/'metadata-trace.so')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
s=(O/'run-metadata.py').read_text().replace("'/tmp/dsv41-hc-on-long-indexer-20261103/metadata'","str(P/'metadata')").replace("LLAMA_DSV41_CANDIDATE_POOL_LOCAL=flag","LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=flag")
(P/'run-metadata.py').write_text(s)
