#include "counts.h"
#include "ggml-backend.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
static std::atomic<bool> active{false};
static std::atomic<uint64_t> counts[counter_count];
template<class F> static F next(const char *name){auto *p=dlsym(RTLD_NEXT,name);if(!p){fprintf(stderr,"COUNTER missing %s\n",name);abort();}return reinterpret_cast<F>(p);}
static void add(int i,uint64_t n=1){if(active.load(std::memory_order_relaxed))counts[i].fetch_add(n,std::memory_order_relaxed);}
extern "C" void dsv_width_count_begin(){active.store(false);for(auto &x:counts)x.store(0);active.store(true);}
extern "C" void dsv_width_count_end(uint64_t *out){active.store(false);for(int i=0;i<counter_count;++i)out[i]=counts[i].load();}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,ggml_cgraph *g){static auto f=next<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");if(active.load(std::memory_order_relaxed)){add(0);int n=ggml_graph_n_nodes(g);add(1,n);for(int i=0;i<n;++i){auto *t=ggml_graph_node(g,i);if(t->op==GGML_OP_MUL_MAT){bool q=ggml_is_quantized(t->src[0]->type);add(q?17:2);if(!q&&t->src[1]->ne[1]==1)add(3);}else if(t->op==GGML_OP_MUL_MAT_ID)add(4);else if(t->op==GGML_OP_GET_ROWS)add(5);}}return f(b,g);}
extern "C" void ggml_backend_synchronize(ggml_backend_t b){static auto f=next<decltype(&ggml_backend_synchronize)>("ggml_backend_synchronize");add(6);f(b);}
extern "C" void ggml_backend_sched_synchronize(ggml_backend_sched_t s){static auto f=next<decltype(&ggml_backend_sched_synchronize)>("ggml_backend_sched_synchronize");add(7);f(s);}
extern "C" void ggml_backend_event_synchronize(ggml_backend_event_t e){static auto f=next<decltype(&ggml_backend_event_synchronize)>("ggml_backend_event_synchronize");add(8);f(e);}
extern "C" void ggml_backend_event_wait(ggml_backend_t b,ggml_backend_event_t e){static auto f=next<decltype(&ggml_backend_event_wait)>("ggml_backend_event_wait");add(9);f(b,e);}
extern "C" void ggml_backend_tensor_copy_async(ggml_backend_t a,ggml_backend_t b,const ggml_tensor *s,ggml_tensor *d){static auto f=next<decltype(&ggml_backend_tensor_copy_async)>("ggml_backend_tensor_copy_async");add(10);f(a,b,s,d);}
extern "C" void ggml_backend_tensor_copy(const ggml_tensor *s,ggml_tensor *d){static auto f=next<decltype(&ggml_backend_tensor_copy)>("ggml_backend_tensor_copy");add(11);f(s,d);}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g,cudaStream_t s){static auto f=next<decltype(&cudaGraphLaunch)>("cudaGraphLaunch");add(12);return f(g,s);}
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t *e,cudaGraph_t g,unsigned long long flags){static auto f=next<decltype(&cudaGraphInstantiate)>("cudaGraphInstantiate");add(13);return f(e,g,flags);}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *c,const void *f,void **a){static auto fn=next<decltype(&cudaLaunchKernelExC)>("cudaLaunchKernelExC");add(14);return fn(c,f,a);}
extern "C" void ggml_backend_sched_prepare_inputs(ggml_backend_sched_t s){static auto f=next<decltype(&ggml_backend_sched_prepare_inputs)>("ggml_backend_sched_prepare_inputs");add(15);f(s);}
extern "C" bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t s,ggml_cgraph *g){static auto f=next<decltype(&ggml_backend_sched_alloc_graph)>("ggml_backend_sched_alloc_graph");add(16);return f(s,g);}
