#include "llama-context.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "speculative.h"
#include "ggml-backend-impl.h"
#include "nlohmann/json.hpp"
#include <dlfcn.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <vector>
#include <unordered_set>
using json = nlohmann::ordered_json;
static int64_t now_ns() { timespec t{}; clock_gettime(CLOCK_MONOTONIC,&t); return int64_t(t.tv_sec)*1000000000+t.tv_nsec; }
template<class F> static F next_fn(const char * n) { auto * p=dlsym(RTLD_NEXT,n); if(!p){fprintf(stderr,"RESIDUAL missing %s\n",n);abort();}return reinterpret_cast<F>(p); }
static int armed() {const char *p=getenv("DSV_RESIDUAL_ARM");int id=0;if(p)if(FILE*f=fopen(p,"r")){if(fscanf(f,"%d",&id)!=1)id=0;fclose(f);}return id;}
struct Scope;
static thread_local Scope * current=nullptr;
static std::atomic<uint64_t> sequence{0};
static std::mutex log_mutex;
struct Scope {
 Scope * parent; uint64_t id; int case_id,depth=0; const char * kind; int pos,tokens,kv;int64_t start;json calls=json::array();
 Scope(const char*k,int c,int p=-1,int n=0,int v=-1):parent(current),id(++sequence),case_id(c),kind(k),pos(p),tokens(n),kv(v),start(now_ns()){if(c)current=this;}
 ~Scope(){if(!case_id)return;int64_t end=now_ns();current=parent;json row={{"id",id},{"parent",parent?parent->id:0},{"case",case_id},{"kind",kind},{"pos",pos},{"tokens",tokens},{"kv_only",kv},{"start",start},{"end",end},{"calls",std::move(calls)}};std::string s=row.dump()+"\n";std::lock_guard<std::mutex> lock(log_mutex);static FILE*f=fopen(getenv("DSV_RESIDUAL_LOG"),"a");if(!f)abort();fwrite(s.data(),1,s.size(),f);fflush(f);}
};
struct Call {
 Scope*s;int64_t start=0;json j;
 Call(const char*k,const void*pc,ggml_backend_t b=nullptr):s(current){if(!s)return;start=now_ns();Dl_info d{};dladdr(pc,&d);j={{"kind",k},{"depth",s->depth++},{"module",d.dli_fname?d.dli_fname:""},{"offset",uintptr_t(pc)-uintptr_t(d.dli_fbase)}};if(b)j["backend"]=ggml_backend_name(b);}
 ~Call(){if(!s)return;--s->depth;j["start"]=start;j["end"]=now_ns();s->calls.push_back(std::move(j));}
};
static json tensor_info(const ggml_tensor*t) {
 json j={{"id",uintptr_t(t)},{"name",t->name},{"bytes",ggml_nbytes(t)},{"data",uintptr_t(t->data)},{"buffer",uintptr_t(t->buffer)},{"type",ggml_type_name(t->type)},{"op",ggml_op_name(t->op)},{"flags",t->flags},{"input",bool(t->flags&GGML_TENSOR_FLAG_INPUT)},{"output",bool(t->flags&GGML_TENSOR_FLAG_OUTPUT)},{"view_src",uintptr_t(t->view_src)},{"view_offs",t->view_offs},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"nb",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}}};
 const ggml_tensor*root=t;std::unordered_set<const ggml_tensor*>path;size_t off=0;
 while(root->view_src){if(!path.insert(root).second){j["view_cycle"]=true;break;}off+=root->view_offs;root=root->view_src;}
 j["root"]=uintptr_t(root);j["root_name"]=root->name;j["root_data"]=uintptr_t(root->data);j["root_bytes"]=ggml_nbytes(root);j["root_offset"]=off;j["root_buffer"]=uintptr_t(root->buffer);
 json src=json::array();for(auto*x:t->src)src.push_back(uintptr_t(x));j["src"]=src;
 json op=json::array();for(auto x:t->op_params)op.push_back(x);j["op_params_i32"]=op;
 if(root->buffer){auto*buf=root->buffer;auto*bt=ggml_backend_buffer_get_type(buf);auto*dev=ggml_backend_buft_get_device(bt);j["buffer_name"]=ggml_backend_buffer_name(buf);j["buffer_type"]=ggml_backend_buft_name(bt);j["buffer_owner"]=dev?ggml_backend_dev_name(dev):"unknown";j["buffer_host"]=ggml_backend_buffer_is_host(buf);j["buffer_usage"]=int(ggml_backend_buffer_get_usage(buf));j["buffer_size"]=ggml_backend_buffer_get_size(buf);j["buffer_meta"]=ggml_backend_buffer_is_meta(buf);j["buffer_base"]=ggml_backend_buffer_is_meta(buf)?0:uintptr_t(ggml_backend_buffer_get_base(buf));}
 return j;
}
static json graph_description(ggml_cgraph*g){
 json out={{"nodes",json::array()},{"tensors",json::array()}};std::vector<const ggml_tensor*>queue;std::unordered_set<const ggml_tensor*>seen;
 for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);out["nodes"].push_back(uintptr_t(t));queue.push_back(t);}
 for(size_t i=0;i<queue.size();++i){auto*t=queue[i];if(!t||!seen.insert(t).second)continue;out["tensors"].push_back(tensor_info(t));if(t->view_src)queue.push_back(t->view_src);for(auto*x:t->src)if(x)queue.push_back(x);}
 return out;
}
extern "C" const char*dsv_owner_describe_graph(ggml_cgraph*g){static thread_local std::string out;out=graph_description(g).dump();return out.c_str();}
static bool describe_current(){
 if(!current||strcmp(current->kind,"ubatch"))return false;auto*s=current;while(s->parent)s=s->parent;if(strcmp(s->kind,"target_decode"))return false;
 const int pos=current->pos;return pos==0||pos==512||pos==1536||pos==2048||pos==4096||pos==4608||pos==6144||pos==7168||pos==7676;
}

using Copy=bool(*)(ggml_backend_t,ggml_backend_t,const ggml_tensor*,ggml_tensor*);
static std::mutex copies_mutex;
static std::unordered_map<ggml_backend_t,Copy> copies;
static bool copy_iface(ggml_backend_t src_b,ggml_backend_t dst_b,const ggml_tensor*src,ggml_tensor*dst){Copy fn;{std::lock_guard<std::mutex> lock(copies_mutex);fn=copies.at(dst_b);}if(!current)return fn(src_b,dst_b,src,dst);Call c("copy_iface",__builtin_return_address(0),dst_b);c.j["source_backend"]=ggml_backend_name(src_b);c.j["src"]=tensor_info(src);c.j["dst"]=tensor_info(dst);bool ok=fn(src_b,dst_b,src,dst);c.j["async_accepted"]=ok;return ok;}
extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t dev,const char*params){static auto fn=next_fn<decltype(&ggml_backend_dev_init)>("ggml_backend_dev_init");auto*b=fn(dev,params);if(b&&b->iface.cpy_tensor_async){std::lock_guard<std::mutex> lock(copies_mutex);copies[b]=b->iface.cpy_tensor_async;b->iface.cpy_tensor_async=copy_iface;}return b;}
int llama_context::decode(const llama_batch&b){using F=int(*)(llama_context*,const llama_batch&);static auto fn=next_fn<F>("_ZN13llama_context6decodeERK11llama_batch");char arch[64]{};llama_model_meta_val_str(llama_get_model(this),"general.architecture",arch,sizeof arch);int c=b.n_tokens>6?armed():0;Scope s(!strcmp(arch,"deepseek41")?"target_decode":"draft_decode",c,b.pos?b.pos[0]:-1,b.n_tokens);return fn(this,b);}
llm_graph_result*llama_context::process_ubatch(const llama_ubatch&b,llm_graph_type gt,llama_memory_context_i*m,ggml_status&r,int kv){using F=llm_graph_result*(*)(llama_context*,const llama_ubatch&,llm_graph_type,llama_memory_context_i*,ggml_status&,int);static auto fn=next_fn<F>("_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_statusi");Scope s("ubatch",current?current->case_id:0,b.pos?b.pos[0]:-1,b.n_tokens,kv);return fn(this,b,gt,m,r,kv);}
bool llama_context::extract_layer_inputs(const llm_graph_result*r,size_t p,size_t n){using F=bool(*)(llama_context*,const llm_graph_result*,size_t,size_t);static auto fn=next_fn<F>("_ZN13llama_context20extract_layer_inputsEPK16llm_graph_resultmm");Scope s("extract_taps",current?current->case_id:0,p,n);return fn(this,r,p,n);}
ggml_tensor*llama_context::get_embeddings_layer_inp_packed(uint32_t row,uint32_t n){using F=ggml_tensor*(*)(llama_context*,uint32_t,uint32_t);static auto fn=next_fn<F>("_ZN13llama_context31get_embeddings_layer_inp_packedEjj");Scope s("get_packet",current?current->case_id:(n>6?armed():0),row,n);return fn(this,row,n);}
bool common_speculative_process(common_speculative*s,const llama_batch&b){using F=bool(*)(common_speculative*,const llama_batch&);static auto fn=next_fn<F>("_Z26common_speculative_processP18common_speculativeRK11llama_batch");Scope range("feature_bridge",b.n_tokens>6?armed():0,b.pos?b.pos[0]:-1,b.n_tokens);return fn(s,b);}
extern "C" void ggml_backend_event_synchronize(ggml_backend_event_t e){static auto fn=next_fn<decltype(&ggml_backend_event_synchronize)>("ggml_backend_event_synchronize");if(!current){fn(e);return;}Call c("event_sync",__builtin_return_address(0));c.j["event"]=uintptr_t(e);c.j["event_device"]=e->device?ggml_backend_dev_name(e->device):"null";fn(e);}
extern "C" void ggml_backend_synchronize(ggml_backend_t b){static auto fn=next_fn<decltype(&ggml_backend_synchronize)>("ggml_backend_synchronize");if(!current){fn(b);return;}Call c("backend_sync",__builtin_return_address(0),b);fn(b);}
extern "C" void ggml_backend_event_wait(ggml_backend_t b,ggml_backend_event_t e){static auto fn=next_fn<decltype(&ggml_backend_event_wait)>("ggml_backend_event_wait");if(!current){fn(b,e);return;}Call c("event_wait",__builtin_return_address(0),b);c.j["event"]=uintptr_t(e);c.j["event_device"]=e->device?ggml_backend_dev_name(e->device):"null";fn(b,e);}
extern "C" bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t s,ggml_cgraph*g){static auto fn=next_fn<decltype(&ggml_backend_sched_alloc_graph)>("ggml_backend_sched_alloc_graph");if(!current)return fn(s,g);Call c("allocate_graph",__builtin_return_address(0));bool ok=fn(s,g);c.j["success"]=ok;c.j["scheduler"]=uintptr_t(s);c.j["copies"]=ggml_backend_sched_get_n_copies(s);if(ok&&describe_current())c.j["allocated_graph"]=graph_description(g);return ok;}
extern "C" void ggml_backend_sched_prepare_inputs(ggml_backend_sched_t s){static auto fn=next_fn<decltype(&ggml_backend_sched_prepare_inputs)>("ggml_backend_sched_prepare_inputs");if(!current){fn(s);return;}Call c("prepare_inputs",__builtin_return_address(0));fn(s);}
extern "C" void ggml_backend_sched_synchronize(ggml_backend_sched_t s){static auto fn=next_fn<decltype(&ggml_backend_sched_synchronize)>("ggml_backend_sched_synchronize");if(!current){fn(s);return;}Call c("scheduler_sync",__builtin_return_address(0));fn(s);}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,ggml_cgraph*g){static auto fn=next_fn<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");if(!current)return fn(b,g);Call c("graph_compute",__builtin_return_address(0),b);c.j["nodes"]=ggml_graph_n_nodes(g);if(ggml_graph_n_nodes(g)){auto*t=ggml_graph_node(g,0);c.j["first"]=tensor_info(t);c.j["last"]=tensor_info(ggml_graph_node(g,ggml_graph_n_nodes(g)-1));}if(describe_current())c.j["graph_snapshot"]=graph_description(g);auto rc=fn(b,g);c.j["status"]=int(rc);return rc;}
extern "C" void ggml_backend_tensor_copy(const ggml_tensor*src,ggml_tensor*dst){static auto fn=next_fn<decltype(&ggml_backend_tensor_copy)>("ggml_backend_tensor_copy");if(!current){fn(src,dst);return;}Call c("copy_sync",__builtin_return_address(0));c.j["src"]=tensor_info(src);c.j["dst"]=tensor_info(dst);fn(src,dst);}
extern "C" void ggml_backend_tensor_copy_async(ggml_backend_t a,ggml_backend_t b,const ggml_tensor*src,ggml_tensor*dst){static auto fn=next_fn<decltype(&ggml_backend_tensor_copy_async)>("ggml_backend_tensor_copy_async");if(!current){fn(a,b,src,dst);return;}Call c("copy_async_api",__builtin_return_address(0),b);c.j["source_backend"]=ggml_backend_name(a);c.j["src"]=tensor_info(src);c.j["dst"]=tensor_info(dst);fn(a,b,src,dst);}

extern "C" void ggml_backend_event_record(ggml_backend_event_t e,ggml_backend_t b){static auto fn=next_fn<decltype(&ggml_backend_event_record)>("ggml_backend_event_record");if(!current){fn(e,b);return;}Call c("event_record",__builtin_return_address(0),b);c.j["event"]=uintptr_t(e);c.j["event_device"]=e->device?ggml_backend_dev_name(e->device):"null";fn(e,b);}

static void reserve_log(const char*kind,ggml_backend_sched_t s,ggml_cgraph*g) {
 const char*p=getenv("DSV_PREFILL_RESERVE_LOG");if(!p)return;
 json j={{"kind",kind},{"copies",ggml_backend_sched_get_n_copies(s)},{"graph",graph_description(g)}};
 auto text=j.dump()+"\n";std::lock_guard<std::mutex>lock(log_mutex);FILE*f=fopen(p,"a");if(!f)abort();fwrite(text.data(),1,text.size(),f);fclose(f);
}
extern "C" void ggml_backend_sched_reserve_size(ggml_backend_sched_t s,ggml_cgraph*g,size_t*sizes){static auto fn=next_fn<decltype(&ggml_backend_sched_reserve_size)>("ggml_backend_sched_reserve_size");reserve_log("reserve_size_before",s,g);fn(s,g,sizes);}
extern "C" bool ggml_backend_sched_reserve(ggml_backend_sched_t s,ggml_cgraph*g){static auto fn=next_fn<decltype(&ggml_backend_sched_reserve)>("ggml_backend_sched_reserve");reserve_log("reserve_before",s,g);return fn(s,g);}
