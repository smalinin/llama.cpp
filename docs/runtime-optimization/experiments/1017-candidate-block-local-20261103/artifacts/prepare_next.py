from pathlib import Path
import json,subprocess,shutil
P=Path(__file__).parent;O=Path('/tmp/dsv41-candidate-pool-20261103');R=Path('/home/sergei/Github/llama.cpp')
s=(O/'bench.py').read_text();start=s.index("cfg=json.load");end=s.index("cmd=cfg",start)
s=s[:start]+'''cfg=json.load(open(P/'config.json'));assert json.load(open(P/'precision-results.json'))['complete'];assert json.load(open(P/'operator-results.json'))['complete']
'''+s[end:]
s=s.replace('candidate-pool-screen','candidate-block-screen').replace('19067','19077').replace('LLAMA_DSV41_CANDIDATE_POOL_LOCAL','LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL').replace('candidate-pool OFF/ON/OFF','candidate-block OFF/ON/OFF; previous pool=1 in all arms');(P/'bench.py').write_text(s)
s=(O/'transfer-audit.cpp').read_text().replace('LLAMA_DSV41_CANDIDATE_POOL_LOCAL','LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL')
s=s.replace('return t && (std::strstr(t->name,"idx_score-20") || std::strstr(t->name,"candidate_"));','return t && (std::strstr(t->name,"idx_score-20") || std::strstr(t->name,"candidate_") || (t->op==GGML_OP_RESHAPE && t->src[0] && std::strstr(t->src[0]->name,"candidate_keep")));')
s=s.replace('for(auto*s:t->src)if(selected(s))tensors.insert(s);','for(auto*s:t->src)if(selected(s) || (s && t->op==GGML_OP_REPEAT && std::strstr(t->name,"candidate_local_repeat")))tensors.insert(s);')
(P/'transfer-audit.cpp').write_text(s);shutil.copyfile(O/'placement.cpp',P/'placement.cpp')
base=['g++','-std=c++17','-O2','-shared','-fPIC','-Iinclude','-Isrc','-Iggml/include','-Ivendor','-I/usr/local/cuda/include','-ldl']
with(P/'compile-probes.log').open('wb')as f:
 for name in ['placement','transfer-audit']:subprocess.run(base+[str(P/(name+'.cpp')),'-o',str(P/(name+'.so'))],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
s=(O/'admit.py').read_text().replace("'/tmp/dsv41-hc-on-long-indexer-20261103/payload-176128/results.json'","'/tmp/dsv41-candidate-pool-20261103/payload-176128/results.json'")
s=s.replace('prior HC-sharing ON/candidateOFF176128','prior HC-sharing ON/poolON176128');(P/'admit.py').write_text(s)
s=(O/'run-payload.py').read_text().replace("env.update(LLAMA_DSV41_CANDIDATE_POOL_LOCAL='1',","env.update(LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL='1',")
s=s.replace('19068','19078').replace('candidate-local-large-context','candidate-block-large-context').replace('candidate-pool ON/HCsharingON','candidate-block ON/poolON/HCsharingON');(P/'run-payload.py').write_text(s)
