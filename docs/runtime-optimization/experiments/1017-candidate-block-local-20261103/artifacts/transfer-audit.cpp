#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <atomic>

using json = nlohmann::ordered_json;
namespace {
std::atomic<uint64_t> serial{0};
std::mutex log_lock;
thread_local bool logging = false;
struct Span {ggml_tensor * t; uintptr_t data; size_t bytes;};
struct Frame {uint64_t id; std::vector<Span> spans; uint32_t calls=0;};
thread_local Frame * active = nullptr;
uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
template<class F> F next(const char * n){auto p=dlsym(RTLD_NEXT,n);if(!p)std::abort();return reinterpret_cast<F>(p);}
void emit(const json & j){
 if(logging)return;
 logging=true;
 const char * path=std::getenv("DSV_TRANSFER_FILE");
 if(path){std::lock_guard<std::mutex>g(log_lock);if(FILE*f=std::fopen(path,"a")){auto s=j.dump();std::fprintf(f,"%s\n",s.c_str());std::fclose(f);}}
 logging=false;
}
std::string addr(const void * p){char s[32];std::snprintf(s,sizeof s,"0x%llx",static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));return s;}
bool selected(const ggml_tensor * t){return t && (std::strstr(t->name,"idx_score-20") || std::strstr(t->name,"candidate_") || (t->op==GGML_OP_RESHAPE && t->src[0] && std::strstr(t->src[0]->name,"candidate_keep")));}
json meta(ggml_tensor * t){
 auto * root=t;
 while(root->view_src)root=root->view_src;
 auto * b=t->buffer?t->buffer:root->buffer;
 json r={{"name",t->name},{"tensor",addr(t)},{"data",addr(t->data)},{"flags",t->flags},{"type",ggml_type_name(t->type)},{"op",ggml_op_name(t->op)},{"bytes",ggml_nbytes(t)},{"contiguous",ggml_is_contiguous(t)},{"view_src",addr(t->view_src)},{"view_offset",t->view_offs},{"root",addr(root)},{"root_data",addr(root->data)},{"root_name",root->name},{"buffer",b?ggml_backend_buffer_name(b):"none"},{"ne",json::array()},{"nb",json::array()}};
 for(int i=0;i<GGML_MAX_DIMS;++i){r["ne"].push_back(t->ne[i]);r["nb"].push_back(t->nb[i]);}
 return r;
}
json matches(const void * p,size_t bytes){
 json ids=json::array();if(!active || !p)return ids;auto q=reinterpret_cast<uintptr_t>(p);
 for(size_t i=0;i<active->spans.size();++i){auto&s=active->spans[i];if(q==s.data && bytes==s.bytes)ids.push_back(i);}
 return ids;
}
}

extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched,ggml_cgraph * graph){
 using Fn=ggml_status(*)(ggml_backend_sched_t,ggml_cgraph*);static auto fn=next<Fn>("ggml_backend_sched_graph_compute_async");
 const char * arm=std::getenv("DSV_TRANSFER_ARM");
 if(!arm || access(arm,F_OK)!=0 || active)return fn(sched,graph);
 std::set<ggml_tensor*> tensors;
 bool wide=false;
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*t=ggml_graph_node(graph,i);if(selected(t)){tensors.insert(t);wide |= t->ne[1]>6;}for(auto*s:t->src)if(selected(s) || (s && t->op==GGML_OP_REPEAT && std::strstr(t->name,"candidate_local_repeat")))tensors.insert(s);}
 if(!wide)return fn(sched,graph);
 Frame frame{++serial,{}};json snapshot={{"event","frame"},{"id",frame.id},{"begin_ns",ns()},{"flag",std::getenv("LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL")?std::getenv("LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL"):"unset"},{"tensors",json::array()},{"consumers",json::array()}};
 for(auto*t:tensors){snapshot["tensors"].push_back(meta(t));frame.spans.push_back({t,reinterpret_cast<uintptr_t>(t->data),ggml_nbytes(t)});}
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*t=ggml_graph_node(graph,i);for(int k=0;k<GGML_MAX_SRC;++k)if(tensors.count(t->src[k]))snapshot["consumers"].push_back({{"node",t->name},{"op",ggml_op_name(t->op)},{"input",k},{"tensor",addr(t->src[k])},{"output_data",addr(t->data)}});}
 emit(snapshot);active=&frame;auto status=fn(sched,graph);active=nullptr;
 json end={{"event","frame_end"},{"id",frame.id},{"end_ns",ns()},{"status",int(status)},{"calls",frame.calls},{"changed_metadata",json::array()}};
 for(auto&s:frame.spans)if(reinterpret_cast<uintptr_t>(s.t->data)!=s.data || ggml_nbytes(s.t)!=s.bytes)end["changed_metadata"].push_back(meta(s.t));
 emit(end);return status;
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void * dst,int dst_dev,const void * src,int src_dev,size_t bytes,cudaStream_t stream){
 using Fn=cudaError_t(*)(void*,int,const void*,int,size_t,cudaStream_t);static auto fn=next<Fn>("cudaMemcpyPeerAsync");
 if(!active || logging)return fn(dst,dst_dev,src,src_dev,bytes,stream);
 auto sm=matches(src,bytes),dm=matches(dst,bytes);auto start=ns();auto rc=fn(dst,dst_dev,src,src_dev,bytes,stream);
 if(!sm.empty() || !dm.empty()){++active->calls;emit({{"event","copy"},{"api","cudaMemcpyPeerAsync"},{"id",active->id},{"seq",active->calls},{"begin_ns",start},{"end_ns",ns()},{"src",addr(src)},{"dst",addr(dst)},{"src_device",src_dev},{"dst_device",dst_dev},{"bytes",bytes},{"stream",addr(stream)},{"source_match",sm},{"destination_match",dm},{"return",int(rc)}});}
 return rc;
}

extern "C" cudaError_t cudaMemcpyAsync(void * dst,const void * src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream){
 using Fn=cudaError_t(*)(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t);static auto fn=next<Fn>("cudaMemcpyAsync");
 if(!active || logging)return fn(dst,src,bytes,kind,stream);
 auto sm=matches(src,bytes),dm=matches(dst,bytes);auto start=ns();auto rc=fn(dst,src,bytes,kind,stream);
 if(!sm.empty() || !dm.empty()){++active->calls;emit({{"event","copy"},{"api","cudaMemcpyAsync"},{"id",active->id},{"seq",active->calls},{"begin_ns",start},{"end_ns",ns()},{"src",addr(src)},{"dst",addr(dst)},{"kind",int(kind)},{"bytes",bytes},{"stream",addr(stream)},{"source_match",sm},{"destination_match",dm},{"return",int(rc)}});}
 return rc;
}
