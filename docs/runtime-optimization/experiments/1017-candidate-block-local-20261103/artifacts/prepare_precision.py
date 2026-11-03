from pathlib import Path
import subprocess
P=Path(__file__).parent;O=Path('/tmp/dsv41-candidate-pool-20261103');R=Path('/home/sergei/Github/llama.cpp')
(P/'fresh.cpp').write_text((O/'fresh-on.cpp').read_text())
s=(O/'trace.cpp').read_text().replace('static long long aliases[3]={}', 'static long long blocks[2]={}, repeats[2]={},dense[2]={};\nstatic long long aliases[3]={}')
s=s.replace(' bool fail=false;', ''' const char* bl=getenv("LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL");int bm=bl&&!strcmp(bl,"1");long long nr=0,nd=0;for(int i=0;i<ggml_graph_n_nodes(gf);++i){auto*n=ggml_graph_node(gf,i);if(!strncmp(n->name,"candidate_local_repeat-",23)&&n->op==GGML_OP_REPEAT)++nr;if(!strcmp(n->name,"candidate_mask-20")&&n->ne[1]>6)++nd;} {std::lock_guard<std::mutex>g(lock);blocks[bm]+=nr>0;repeats[bm]+=nr;dense[bm]+=nd;}
 bool fail=false;''')
s=s.replace('std::fclose(f);std::string more=', '''std::fclose(f);std::string bp=std::string(p)+".block.json";f=std::fopen(bp.c_str(),"w");std::fprintf(f,"{\\"active_graphs\\":[%lld,%lld],\\"local_repeats\\":[%lld,%lld],\\"legacy_dense\\":[%lld,%lld]}\\n",blocks[0],blocks[1],repeats[0],repeats[1],dense[0],dense[1]);std::fclose(f);std::string more=''')
(P/'trace.cpp').write_text(s)
base=['g++','-std=c++17','-O2','-Iinclude','-Icommon','-Isrc','-Iggml/include','-Ivendor','-Lbuild-aliceai/bin','-Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin']
with(P/'compile-precision.log').open('wb')as f:
 subprocess.run(base+[str(P/'fresh.cpp'),'-lllama-common','-lllama','-lggml','-lggml-base','-lcrypto','-ldl','-lpthread','-o',str(P/'fresh')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(base+[str(P/'trace.cpp'),'-shared','-fPIC','-ldl','-o',str(P/'trace.so')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
