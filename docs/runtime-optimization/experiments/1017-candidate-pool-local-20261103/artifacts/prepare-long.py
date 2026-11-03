from pathlib import Path
import json,shutil
P=Path(__file__).parent;O=Path('/tmp/dsv41-hc-copy-sharing-20261103')
for n in ['memory_monitor.py','inputs.json','check-long.py']:shutil.copyfile(O/n,P/n)
s=(O/'long.cpp').read_text().replace('GGML_SCHED_COPY_RESHAPE','LLAMA_DSV41_CANDIDATE_POOL_LOCAL');(P/'long.cpp').write_text(s)
s=(O/'trace.cpp').read_text().replace('GGML_SCHED_COPY_RESHAPE','LLAMA_DSV41_CANDIDATE_POOL_LOCAL')
s=s.replace('static long long aliases[3]={};','static long long aliases[3]={},local_pool[3]={},remote_pool[3]={};')
s=s.replace('bool fail=false;', '''long long local=0,remote=0;ggml_tensor*score=nullptr;for(int i=0;i<ggml_graph_n_nodes(gf);++i){auto*n=ggml_graph_node(gf,i);if(!strcmp(n->name,"idx_score-20"))score=n;}for(int i=0;i<ggml_graph_n_nodes(gf);++i){auto*n=ggml_graph_node(gf,i);if(!strcmp(n->name,"candidate_block_score-20")&&n->ne[1]>6&&score&&score->buffer&&n->buffer){if(!strcmp(ggml_backend_buffer_name(score->buffer),ggml_backend_buffer_name(n->buffer)))++local;else++remote;}}
 bool fail=false;''')
s=s.replace('++calls[mode];aliases[mode]+=shared;', '++calls[mode];local_pool[mode]+=local;remote_pool[mode]+=remote;aliases[mode]+=shared;')
s=s.replace('std::fclose(f);}', 'std::fclose(f);}')
# Add machine-readable local/remote counts beside the existing result.
s=s.replace('std::fclose(f);}', '''std::fclose(f);std::string more=std::string(p)+".pool.json";f=std::fopen(more.c_str(),"w");std::fprintf(f,"{\\"local\\":[%lld,%lld,%lld],\\"remote\\":[%lld,%lld,%lld]}\\n",local_pool[0],local_pool[1],local_pool[2],remote_pool[0],remote_pool[1],remote_pool[2]);std::fclose(f);}''')
s=s.replace('#include <algorithm>','#include <algorithm>\n#include <string>');(P/'trace.cpp').write_text(s)
s=(O/'run-long.py').read_text().replace("'scope':'Exact transport OFF/ON/OFF", "'scope':'Exact candidate-pool placement OFF/ON/OFF, HC sharing defaultON")
(P/'run-long.py').write_text(s)
