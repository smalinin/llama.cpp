#include "ggml-backend.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static int remaining=0,failures=0;
static long long calls=0,injections=0;
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t s,ggml_cgraph*g){using F=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static auto fn=reinterpret_cast<F>(dlsym(RTLD_NEXT,"ggml_backend_sched_graph_compute_async"));if(!fn)std::abort();bool inject=false;for(int i=0;i<ggml_graph_n_nodes(g);++i)if(std::strstr(ggml_graph_node(g,i)->name,"kv_injected"))inject=true;++calls;injections+=inject;const char*arm=std::getenv("DSV41_TAIL_FAULT_AFTER");if(!arm)remaining=0;else if(inject){if(!remaining)remaining=std::atoi(arm);if(remaining>0&&!--remaining){++failures;std::fprintf(stderr,"INJECTED_COMPUTE_FAILURE\n");return GGML_STATUS_FAILED;}}return fn(s,g);}
__attribute__((destructor))static void done(){if(const char*p=std::getenv("DSV41_TRACE_FILE")){if(FILE*f=std::fopen(p,"w")){std::fprintf(f,"{\"calls\":%lld,\"injection_graphs\":%lld,\"injected_failures\":%d}\n",calls,injections,failures);std::fclose(f);}}}
