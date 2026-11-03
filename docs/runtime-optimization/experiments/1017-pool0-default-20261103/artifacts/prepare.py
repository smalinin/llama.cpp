from pathlib import Path
import json,hashlib,shutil,subprocess
P=Path(__file__).parent;O=Path('/tmp/dsv41-candidate-block-local-20261103');R=Path('/home/sergei/Github/llama.cpp')
for n in ['memory_monitor.py','inputs.json','metadata.cpp','metadata-trace.cpp','placement.cpp']:shutil.copyfile(O/n,P/n)
cfg=json.load(open(O/'config.json'));cfg['environment'].pop('LLAMA_DSV41_CANDIDATE_POOL_LOCAL',None);cfg['environment'].pop('LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL',None);cfg['source_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();cfg['runtime_patch']=subprocess.check_output(['git','diff','--','src'],cwd=R,text=True);assert not cfg['runtime_patch'];names=list(cfg['source_library_hashes'])+['libllama-server-impl.so','libggml.so'];cfg['source_library_hashes']={n:hashlib.sha256((R/'build-aliceai/bin'/n).read_bytes()).hexdigest()for n in names};cfg['source_files']={n:hashlib.sha256((R/n).read_bytes()).hexdigest()for n in ['src/llama-context.cpp','src/llama-context.h','src/llama-graph.h','src/models/deepseek41.cpp','src/models/models.h']};(P/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
s=(O/'fresh.cpp').read_text();needle='setenv("LLAMA_DSV41_CANDIDATE_POOL_LOCAL","1",1);';assert s.count(needle)==1;s=s.replace(needle,'');assert 'setenv("LLAMA_DSV41_CANDIDATE_POOL_LOCAL"'not in s;(P/'fresh.cpp').write_text(s)
s=(O/'trace.cpp').read_text().replace('int bm=bl&&!strcmp(bl,"1");','int bm=!bl||!strcmp(bl,"1");').replace('else if(mode==1&&inject)','else if(inject)');assert 'mode==1&&inject'not in s;(P/'trace.cpp').write_text(s)
s=(O/'run-precision.py').read_text().replace("for flag in ['0','1']:","for flag in ['0','default']:")
s=s.replace(" root=P/f'precision-{flag}';", " if flag=='default':env.pop('LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL',None)\n assert 'LLAMA_DSV41_CANDIDATE_POOL_LOCAL'not in env\n root=P/f'precision-{flag}';")
s=s.replace('Full two98K corpora/current frozen poolON.','Full two98K corpora/pooling env unset (OFF); block OFF vs actual unset/defaultON.')
s=s.replace(" assert trace['active_graphs'][int(flag)]>0 if flag=='1' else trace['active_graphs']==[0,0]", " assert trace['active_graphs'][1]>0 if flag=='default' else trace['active_graphs']==[0,0]\n pool=json.load(open(P/f'precision-{flag}-trace.json.pool.json'));assert pool['remote'][0]>0 and not any(pool['local']);arm['pooling_placement']=pool")
s=s.replace(" if flag=='1':assert trace['legacy_dense'][1]==0", " if flag=='default':assert trace['legacy_dense'][1]==0")
(P/'run-precision.py').write_text(s)
base=['g++','-std=c++17','-O2','-Iinclude','-Icommon','-Isrc','-Iggml/include','-Ivendor','-Lbuild-aliceai/bin','-Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin']
with(P/'compile.log').open('wb')as f:
 subprocess.run(base+[str(P/'fresh.cpp'),'-lllama-common','-lllama','-lggml','-lggml-base','-lcrypto','-ldl','-lpthread','-o',str(P/'fresh')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(base+[str(P/'metadata.cpp'),'-lllama-common','-lllama','-lggml','-lggml-base','-ldl','-lpthread','-o',str(P/'metadata')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 for name in ['trace','metadata-trace','placement']:subprocess.run(base+[str(P/(name+'.cpp')),'-fPIC','-shared','-ldl','-o',str(P/(name+'.so'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
print('PREPARED frozen current code, pool env unset, block default vs0',flush=True)
