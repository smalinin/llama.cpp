#include "ggml-backend.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <algorithm>
static std::mutex lock;
static long long calls[3]={},masked[3]={},kvonly[3]={},maxcomp[3]={},prefillmasked[3]={};
static int remaining=0,faults=0;
static long long aliases[3]={};
extern "C" enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched,ggml_cgraph*gf){
 using Fn=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static Fn real=(Fn)dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async");if(!real)std::abort();
 const char*e=std::getenv("GGML_SCHED_COPY_RESHAPE");int mode=!e?0:std::strcmp(e,"1")==0?1:2;
 bool ffn=false,pub=false,mask=false,inject=false;long long rows=0,queries=0;
 for(int i=0;i<ggml_graph_n_nodes(gf);++i){auto*t=ggml_graph_node(gf,i);const char*n=t->name;if(std::strstr(n,"kv_injected"))inject=true;if((std::strcmp(n,"ffn_moe_out-20")==0||std::strcmp(n,"ffn_out-20")==0))ffn=true;if(std::strcmp(n,"comp_state_score-20")==0||std::strcmp(n,"comp_state_kv-20")==0)pub=true;if(std::strcmp(n,"candidate_mask-20")==0&&t->data){mask=true;rows=t->ne[0];queries=t->ne[1];}}
 long long shared=0;for(int i=0;i<ggml_graph_n_nodes(gf);++i){auto*n=ggml_graph_node(gf,i);if(n->op==GGML_OP_RMS_NORM&&n->src[0]&&std::strstr(n->src[0]->name,"#hc-view#"))++shared;}
 bool fail=false;
 {std::lock_guard<std::mutex>g(lock);++calls[mode];aliases[mode]+=shared;masked[mode]+=mask;prefillmasked[mode]+=mask&&queries>6;kvonly[mode]+=pub&&!ffn;maxcomp[mode]=std::max(maxcomp[mode],rows);const char*a=std::getenv("DSV41_TAIL_FAULT_AFTER");if(!a){remaining=0;}else if(mode==1&&inject){if(!remaining)remaining=std::atoi(a);if(remaining>0&&!--remaining){++faults;fail=true;}}}
 if(fail){std::fprintf(stderr,"INJECTED_COMPUTE_FAILURE\n");return GGML_STATUS_FAILED;}
 return real(sched,gf);
}
__attribute__((destructor))static void done(){const char*p=std::getenv("DSV41_TRACE_FILE");if(!p)return;FILE*f=std::fopen(p,"w");if(!f)return;std::fprintf(f,"{\"modes\":[");for(int m=0;m<3;++m)std::fprintf(f,"%s{\"mode\":%d,\"calls\":%lld,\"candidate_graphs\":%lld,\"kv_only_graphs\":%lld,\"max_candidate_rows\":%lld,\"prefill_candidate_graphs\":%lld}",m?",":"",m,calls[m],masked[m],kvonly[m],maxcomp[m],prefillmasked[m]);std::fprintf(f,"],\"aliases\":[%lld,%lld,%lld],\"injected_failures\":%d}\n",aliases[0],aliases[1],aliases[2],faults);std::fclose(f);}
