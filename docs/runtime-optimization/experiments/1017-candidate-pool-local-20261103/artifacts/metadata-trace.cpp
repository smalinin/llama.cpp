#include "ggml-backend.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
static std::mutex lock;
void ggml_backend_sched_split_graph(ggml_backend_sched_t s,ggml_cgraph*g){
 using F=void(*)(ggml_backend_sched_t,ggml_cgraph*);static auto real=(F)dlsym(RTLD_NEXT,"ggml_backend_sched_split_graph");if(!real){fprintf(stderr,"missing split symbol\n");abort();}real(s,g);auto*path=getenv("DSV41_META_TRACE");if(!path)return;std::lock_guard<std::mutex>guard(lock);FILE*f=fopen(path,"a");if(!f)abort();for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(strcmp(t->name,"idx_score-20")&&strncmp(t->name,"candidate_",10))continue;auto*b=ggml_backend_sched_get_tensor_backend(s,t);fprintf(f,"{\"name\":\"%s\",\"op\":\"%s\",\"backend\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld]}\n",t->name,ggml_op_name(t->op),b?ggml_backend_name(b):"none",(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3]);}fclose(f);
}
