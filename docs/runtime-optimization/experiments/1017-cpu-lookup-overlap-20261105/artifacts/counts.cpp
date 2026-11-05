#include "llama-context.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "ggml-backend-impl.h"
#include <dlfcn.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <cstdint>

static int64_t ns() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000000000LL+t.tv_nsec; }
template<class F>static F next(const char*n){void*p=dlsym(RTLD_NEXT,n);if(!p)abort();return reinterpret_cast<F>(p);}
static std::string quote(const char*s){std::string v="\"";for(;s&&*s;++s){if(*s=='"'||*s=='\\')v+='\\';if(*s=='\n')v+="\\n";else v+=*s;}return v+'"';}
struct Count{int count=0;int64_t time=0;};
struct Scope{int id=0,pos=0,tokens=0,kv=-1;std::map<uintptr_t,Count> wait;std::vector<std::string> cpu;};
static thread_local Scope* scope=nullptr;
static thread_local int target=0,case_id=0;
static FILE* log_file(){static FILE*f=[](){const char*p=getenv("DSV_COUNT_LOG");return p?fopen(p,"a"):nullptr;}();return f;}
int llama_context::decode(const llama_batch&b){using F=int(*)(llama_context*,const llama_batch&);static auto fn=next<F>("_ZN13llama_context6decodeERK11llama_batch");char arch[64]{};llama_model_meta_val_str(llama_get_model(this),"general.architecture",arch,sizeof arch);int old=target,oldid=case_id;target=!strcmp(arch,"deepseek41");case_id=0;if(target&&b.n_tokens>6){if(auto*p=getenv("DSV_COUNT_ARM")){if(FILE*f=fopen(p,"r")){if(fscanf(f,"%d",&case_id)!=1)case_id=0;fclose(f);}}}int r=fn(this,b);target=old;case_id=oldid;return r;}
llm_graph_result*llama_context::process_ubatch(const llama_ubatch&b,llm_graph_type gt,llama_memory_context_i*m,ggml_status&s,int kv){using F=llm_graph_result*(*)(llama_context*,const llama_ubatch&,llm_graph_type,llama_memory_context_i*,ggml_status&,int);static auto fn=next<F>("_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_statusi");if(!target||!case_id)return fn(this,b,gt,m,s,kv);Scope v;v.id=case_id;v.pos=b.pos?b.pos[0]:-1;v.tokens=b.n_tokens;v.kv=kv;Scope*old=scope;scope=&v;auto a=ns();auto*r=fn(this,b,gt,m,s,kv);auto z=ns();scope=old;if(FILE*f=log_file()){fprintf(f,"{\"case\":%d,\"pos\":%d,\"tokens\":%d,\"kv_only\":%d,\"start\":%lld,\"end\":%lld,\"waits\":[",v.id,v.pos,v.tokens,v.kv,(long long)a,(long long)z);bool first=true;for(auto&[pc,c]:v.wait){Dl_info d{};dladdr((void*)pc,&d);fprintf(f,"%s{\"module\":%s,\"offset\":%llu,\"calls\":%d,\"ns\":%lld}",first?"":",",quote(d.dli_fname).c_str(),(unsigned long long)(pc-(uintptr_t)d.dli_fbase),c.count,(long long)c.time);first=false;}fprintf(f,"],\"cpu_graphs\":[");for(size_t i=0;i<v.cpu.size();++i)fprintf(f,"%s%s",i?",":"",v.cpu[i].c_str());fprintf(f,"]}\n");fflush(f);}return r;}
extern "C" void ggml_backend_event_synchronize(ggml_backend_event_t e){static auto fn=next<decltype(&ggml_backend_event_synchronize)>("ggml_backend_event_synchronize");if(!scope){fn(e);return;}auto a=ns();fn(e);auto&c=scope->wait[(uintptr_t)__builtin_return_address(0)];c.count++;c.time+=ns()-a;}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,ggml_cgraph*g){static auto fn=next<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");if(!scope||ggml_backend_dev_type(ggml_backend_get_device(b))!=GGML_BACKEND_DEVICE_TYPE_CPU)return fn(b,g);auto a=ns();auto ret=fn(b,g);auto z=ns();std::string v="{\"start\":"+std::to_string(a)+",\"end\":"+std::to_string(z)+",\"nodes\":[";for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(i)v+=',';v+="{\"name\":"+quote(t->name)+",\"op\":"+quote(ggml_op_name(t->op))+",\"data\":"+std::to_string((uintptr_t)t->data)+",\"buffer\":"+std::to_string((uintptr_t)t->buffer)+",\"flags\":"+std::to_string(t->flags);if(t->src[0])v+=",\"src0\":"+quote(t->src[0]->name);if(t->src[1])v+=",\"src1\":"+quote(t->src[1]->name)+",\"src1_flags\":"+std::to_string(t->src[1]->flags)+",\"src1_op\":"+quote(ggml_op_name(t->src[1]->op));v+='}';}scope->cpu.push_back(v+"]}");return ret;}
