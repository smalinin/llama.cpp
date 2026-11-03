#include "llama-context.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-ext.h"
#include "speculative.h"
#include "ggml-backend.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <unistd.h>
#include <string>

namespace {
long long clock_ns(){timespec t{};clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1000000000LL+t.tv_nsec;}
template<class F>F next(const char*n){void*p=dlsym(RTLD_NEXT,n);if(!p){fprintf(stderr,"PROFILE missing %s: %s\n",n,dlerror());abort();}return reinterpret_cast<F>(p);}
std::string quote(const char*s){std::string r="\"";for(auto*p=(const unsigned char*)s;p&&*p;++p){if(*p=='"'||*p=='\\'){r+='\\';r+=char(*p);}else if(*p<32||*p>=127){char b[8];snprintf(b,sizeof b,"\\u%04x",unsigned(*p));r+=b;}else r+=char(*p);}return r+'"';}
struct State{
 std::atomic<bool> record{false},trace{false},quit{false};std::atomic<int>id{0},end{0};std::mutex log_mutex;FILE*log=nullptr;std::thread watcher;int last_id=0;std::set<uint64_t> cpu_seen;
 State(){if(auto*p=getenv("DSV_PROFILE_EVENTS"))log=fopen(p,"a");}
};
State&S(){static auto*s=new State;return*s;}
thread_local bool feature_get_active=false, pack_graph_active=false, retain_active=false;
thread_local const char*kind="none";thread_local int ubpos=-1,ubtokens=0;thread_local bool inside_bridge=false, prefill_call=false, bridge_prefill=false;
const char*role(){if(retain_active)return prefill_call?"feature_retain":"feature_retain_generation";if(pack_graph_active)return bridge_prefill?"feature_pack":"feature_pack_generation";if(feature_get_active)return bridge_prefill?"feature_wait":"feature_wait_generation";if(!strcmp(kind,"target"))return prefill_call?"target_prefill":"target_generation";if(!strcmp(kind,"draft"))return inside_bridge?(bridge_prefill?"draft_seed":"draft_update"):"draft_generate";return"other";}
void event(const char*type,long long begin,long long end,int pos,int n,const char*name="",size_t bytes=0){auto&s=S();if(!s.record.load()||!s.log)return;std::lock_guard<std::mutex>lock(s.log_mutex);fprintf(s.log,"{\"case\":%d,\"event\":%s,\"role\":%s,\"pos\":%d,\"tokens\":%d,\"begin_ns\":%lld,\"end_ns\":%lld,\"name\":%s,\"bytes\":%zu}\n",s.id.load(),quote(type).c_str(),quote(role()).c_str(),pos,n,begin,end,quote(name).c_str(),bytes);fflush(s.log);}
struct Range{bool on;explicit Range(const char*n):on(S().trace.load()){if(on)nvtxRangePushA(n);}~Range(){if(on)nvtxRangePop();}};
void maybe_arm(int pos){auto&s=S();if(pos!=0||s.record.load())return;const char*p=getenv("DSV_PROFILE_ARM");if(!p)return;FILE*f=fopen(p,"r");if(!f)return;int id=0,tr=0,end=0;int got=fscanf(f,"%d %d %d",&id,&tr,&end);fclose(f);if(got!=3||id<=s.last_id||end<=0)return;s.last_id=id;s.id=id;s.end=end;
 if(tr){auto rc=cudaProfilerStart();if(rc!=cudaSuccess){fprintf(stderr,"PROFILE start failed %d\n",int(rc));abort();}s.trace=true;}
 s.record=true;fprintf(stderr,"PROFILE_ARM case=%d trace=%d prompt=%d\n",id,tr,end);
}
void cpu_meta(ggml_cgraph*g){auto&s=S();if(!s.record.load()||!s.log)return;uint64_t h=1469598103934665603ULL;for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);for(unsigned char c:std::string(t->name)){h^=c;h*=1099511628211ULL;}for(auto n:t->ne){h^=n;h*=1099511628211ULL;}}
 std::lock_guard<std::mutex>lock(s.log_mutex);if(!s.cpu_seen.insert(h).second)return;fprintf(s.log,"{\"event\":\"cpu_metadata\",\"hash\":\"%016llx\",\"nodes\":[",(unsigned long long)h);
 for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);fprintf(s.log,"%s{\"op\":%s,\"name\":%s,\"type\":%s,\"src0\":%s,\"src0_type\":%s,\"ne\":[%lld,%lld,%lld,%lld]}",i?",":"",quote(ggml_op_name(t->op)).c_str(),quote(t->name).c_str(),quote(ggml_type_name(t->type)).c_str(),quote(t->src[0]?t->src[0]->name:"").c_str(),quote(t->src[0]?ggml_type_name(t->src[0]->type):"").c_str(),(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3]);}fprintf(s.log,"]}\n");fflush(s.log);
}
}

int llama_context::decode(const llama_batch&b){using F=int(*)(llama_context*,const llama_batch&);static auto fn=next<F>("_ZN13llama_context6decodeERK11llama_batch");auto*c=this;char arch[64]={};llama_model_meta_val_str(llama_get_model(c),"general.architecture",arch,sizeof arch);const bool target=!strcmp(arch,"deepseek41");int pos=b.pos?b.pos[0]:-1;if(target&&b.token)maybe_arm(pos);auto oldkind=kind;bool oldpref=prefill_call;int oldpos=ubpos,oldtokens=ubtokens;prefill_call=b.pos&&b.n_tokens>0&&b.pos[b.n_tokens-1]<S().end.load();kind=target?"target":!strcmp(arch,"dflash")?"draft":"other";ubpos=pos;ubtokens=b.n_tokens;char label[192];snprintf(label,sizeof label,"decode.%s case=%d pos=%d tokens=%d embd=%d",role(),S().id.load(),pos,b.n_tokens,b.embd!=nullptr);long long start=clock_ns();int rc;{Range range(label);rc=fn(c,b);}event("decode",start,clock_ns(),pos,b.n_tokens);kind=oldkind;prefill_call=oldpref;ubpos=oldpos;ubtokens=oldtokens;return rc;}

llm_graph_result*llama_context::process_ubatch(const llama_ubatch&b,llm_graph_type gt,llama_memory_context_i*m,ggml_status&status,int kv_only_layer){using F=llm_graph_result*(*)(llama_context*,const llama_ubatch&,llm_graph_type,llama_memory_context_i*,ggml_status&,int);static auto fn=next<F>("_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_statusi");ubpos=b.pos?b.pos[0]:-1;ubtokens=b.n_tokens;char label[192];snprintf(label,sizeof label,"ubatch.%s case=%d pos=%d tokens=%u",role(),S().id.load(),ubpos,b.n_tokens);auto start=clock_ns();llm_graph_result*r;{Range range(label);r=fn(this,b,gt,m,status,kv_only_layer);}event("ubatch",start,clock_ns(),ubpos,b.n_tokens);if(kv_only_layer>=0)event("m2_kv_only",start,clock_ns(),ubpos,b.n_tokens);return r;}

bool common_speculative_process(common_speculative*spec,const llama_batch&b){using F=bool(*)(common_speculative*,const llama_batch&);static auto fn=next<F>("_Z26common_speculative_processP18common_speculativeRK11llama_batch");auto oldkind=kind;int oldpos=ubpos,oldtokens=ubtokens;bool oldbridge=inside_bridge,oldpref=prefill_call,oldbp=bridge_prefill;inside_bridge=true;prefill_call=bridge_prefill=b.pos&&b.n_tokens>0&&b.pos[b.n_tokens-1]<S().end.load();kind="target";ubpos=b.pos?b.pos[0]:-1;ubtokens=b.n_tokens;char label[160];snprintf(label,sizeof label,"bridge.%s case=%d pos=%d tokens=%d",bridge_prefill?"prefill":"generation",S().id.load(),ubpos,b.n_tokens);auto start=clock_ns();bool rc;{Range range(label);rc=fn(spec,b);}event("bridge",start,clock_ns(),b.pos?b.pos[0]:-1,b.n_tokens);inside_bridge=oldbridge;prefill_call=oldpref;bridge_prefill=oldbp;kind=oldkind;ubpos=oldpos;ubtokens=oldtokens;return rc;}

float*llama_get_embeddings_layer_inp(llama_context*c,uint32_t lid){static auto fn=next<decltype(&llama_get_embeddings_layer_inp)>("_Z30llama_get_embeddings_layer_inpP13llama_contextj");char label[192];snprintf(label,sizeof label,"feature.get role=feature_wait case=%d layer=%u",S().id.load(),lid);bool old=feature_get_active;feature_get_active=true;auto start=clock_ns();float*r;{Range range(label);r=fn(c,lid);}feature_get_active=old;event("feature_get",start,clock_ns(),ubpos,ubtokens,label);return r;}

ggml_tensor*llama_context::get_embeddings_layer_inp_packed(uint32_t row,uint32_t nrows){using F=ggml_tensor*(*)(llama_context*,uint32_t,uint32_t);static auto fn=next<F>("_ZN13llama_context31get_embeddings_layer_inp_packedEjj");auto*c=this;bool old=feature_get_active;int oldpos=ubpos,oldtokens=ubtokens;feature_get_active=true;ubpos=oldpos+row;ubtokens=nrows;char label[256];snprintf(label,sizeof label,"feature.packed role=feature_wait case=%d pos=%d tokens=%u row=%u",S().id.load(),ubpos,nrows,row);auto start=clock_ns();ggml_tensor*r;{Range range(label);r=fn(c,row,nrows);}feature_get_active=old;event("feature_get_packed",start,clock_ns(),ubpos,nrows,label,r?ggml_nbytes(r):0);ubpos=oldpos;ubtokens=oldtokens;return r;}

extern "C" ggml_status ggml_backend_graph_compute(ggml_backend_t b,ggml_cgraph*g){static auto fn=next<decltype(&ggml_backend_graph_compute)>("ggml_backend_graph_compute");if(!feature_get_active)return fn(b,g);bool old=pack_graph_active;pack_graph_active=true;char label[256];snprintf(label,sizeof label,"pack.compute role=%s case=%d pos=%d tokens=%d backend=%s nodes=%d",role(),S().id.load(),ubpos,ubtokens,ggml_backend_name(b),ggml_graph_n_nodes(g));auto start=clock_ns();ggml_status rc;{Range range(label);rc=fn(b,g);}event("pack_graph",start,clock_ns(),ubpos,ubtokens,label);pack_graph_active=old;return rc;}

static const char*buffer_name(const ggml_tensor*t){auto*b=t->view_src?t->view_src->buffer:t->buffer;return b?ggml_backend_buffer_name(b):"none";}
extern "C" void ggml_backend_tensor_copy_async(ggml_backend_t bs,ggml_backend_t bd,const ggml_tensor*s,ggml_tensor*d){static auto fn=next<decltype(&ggml_backend_tensor_copy_async)>("ggml_backend_tensor_copy_async");bool old=retain_active;retain_active=!strcmp(kind,"target")&&strstr(s->name,"layer_inp")&&strcmp(s->name,d->name)==0;char label[1024];snprintf(label,sizeof label,"copy_async role=%s case=%d pos=%d tokens=%d bytes=%zu src_device=%s dst_device=%s src_buffer=%s dst_buffer=%s src_buf=%p dst_buf=%p tensor=%s|%s",role(),S().id.load(),ubpos,ubtokens,ggml_nbytes(s),ggml_backend_name(bs),ggml_backend_name(bd),buffer_name(s),buffer_name(d),(void*)s->buffer,(void*)d->buffer,s->name,d->name);auto start=clock_ns();{Range range(label);fn(bs,bd,s,d);}event("copy_async",start,clock_ns(),ubpos,ubtokens,label,ggml_nbytes(s));retain_active=old;}
extern "C" void ggml_backend_tensor_copy(const ggml_tensor*s,ggml_tensor*d){static auto fn=next<decltype(&ggml_backend_tensor_copy)>("ggml_backend_tensor_copy");char label[1024];snprintf(label,sizeof label,"copy_sync role=%s case=%d pos=%d tokens=%d bytes=%zu src_buffer=%s dst_buffer=%s src_buf=%p dst_buf=%p tensor=%s|%s",role(),S().id.load(),ubpos,ubtokens,ggml_nbytes(s),buffer_name(s),buffer_name(d),(void*)s->buffer,(void*)d->buffer,s->name,d->name);auto start=clock_ns();{Range range(label);fn(s,d);}event("copy_sync",start,clock_ns(),ubpos,ubtokens,label,ggml_nbytes(s));}
extern "C" void ggml_backend_tensor_set(ggml_tensor*t,const void*d,size_t off,size_t n){static auto fn=next<decltype(&ggml_backend_tensor_set)>("ggml_backend_tensor_set");char label[512];snprintf(label,sizeof label,"upload.sync role=%s case=%d pos=%d tokens=%d bytes=%zu dst_buffer=%s tensor=%s",role(),S().id.load(),ubpos,ubtokens,n,buffer_name(t),t->name);auto start=clock_ns();{Range range(label);fn(t,d,off,n);}event("upload_sync",start,clock_ns(),ubpos,ubtokens,label,n);}
extern "C" void ggml_backend_tensor_get(const ggml_tensor*t,void*d,size_t off,size_t n){static auto fn=next<decltype(&ggml_backend_tensor_get)>("ggml_backend_tensor_get");char label[512];snprintf(label,sizeof label,"download.sync role=%s case=%d pos=%d tokens=%d bytes=%zu src_buffer=%s tensor=%s",role(),S().id.load(),ubpos,ubtokens,n,buffer_name(t),t->name);auto start=clock_ns();{Range range(label);fn(t,d,off,n);}event("download_sync",start,clock_ns(),ubpos,ubtokens,label,n);}

extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,ggml_cgraph*g){static auto fn=next<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");bool cpu=!strcmp(ggml_backend_name(b),"CPU");if(cpu)cpu_meta(g);char label[224];snprintf(label,sizeof label,"backend.%s role=%s case=%d pos=%d tokens=%d nodes=%d",ggml_backend_name(b),role(),S().id.load(),ubpos,ubtokens,ggml_graph_n_nodes(g));auto start=clock_ns();ggml_status r;{Range range(label);r=fn(b,g);}if(cpu)event("cpu_graph",start,clock_ns(),ubpos,ubtokens,label);return r;}
extern "C" void ggml_backend_tensor_get_async(ggml_backend_t b,const ggml_tensor*t,void*d,size_t off,size_t n){static auto fn=next<decltype(&ggml_backend_tensor_get_async)>("ggml_backend_tensor_get_async");char label[256];snprintf(label,sizeof label,"download backend=%s role=%s case=%d pos=%d tokens=%d bytes=%zu tensor=%s",ggml_backend_name(b),role(),S().id.load(),ubpos,ubtokens,n,t->name);auto start=clock_ns();{Range range(label);fn(b,t,d,off,n);}event("download",start,clock_ns(),ubpos,ubtokens,t->name,n);}


extern "C" void ggml_backend_tensor_set_async(ggml_backend_t b,ggml_tensor*t,const void*d,size_t off,size_t n){static auto fn=next<decltype(&ggml_backend_tensor_set_async)>("ggml_backend_tensor_set_async");char label[256];snprintf(label,sizeof label,"upload backend=%s role=%s case=%d pos=%d tokens=%d bytes=%zu tensor=%s",ggml_backend_name(b),role(),S().id.load(),ubpos,ubtokens,n,t->name);auto start=clock_ns();{Range range(label);fn(b,t,d,off,n);}event("upload",start,clock_ns(),ubpos,ubtokens,t->name,n);}


extern "C" size_t llama_state_seq_get_data_ext(llama_context*c,uint8_t*d,size_t size,llama_seq_id seq,llama_state_seq_flags flags){static auto fn=next<decltype(&llama_state_seq_get_data_ext)>("llama_state_seq_get_data_ext");char arch[64]={};llama_model_meta_val_str(llama_get_model(c),"general.architecture",arch,sizeof arch);auto oldkind=kind;bool oldpref=prefill_call;int oldpos=ubpos,oldtokens=ubtokens;prefill_call=(flags&LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)!=0;kind=!strcmp(arch,"deepseek41")?"target":"draft";ubpos=llama_memory_seq_pos_max(llama_get_memory(c),seq);ubtokens=0;char label[160];snprintf(label,sizeof label,"checkpoint.%s case=%d pos=%d bytes=%zu flags=%u",kind,S().id.load(),ubpos,size,flags);auto start=clock_ns();size_t n;{Range range(label);n=fn(c,d,size,seq,flags);}event("checkpoint",start,clock_ns(),ubpos,0,label,n);kind=oldkind;prefill_call=oldpref;ubpos=oldpos;ubtokens=oldtokens;return n;}

__attribute__((constructor))static void setup(){auto&s=S();if(auto*p=getenv("DSV_PROFILE_PID")){if(FILE*f=fopen(p,"w")){fprintf(f,"%d\n",int(getpid()));fclose(f);}}
 s.watcher=std::thread([]{auto&s=S();int finished=0;while(!s.quit.load()){std::this_thread::sleep_for(std::chrono::milliseconds(20));auto*p=getenv("DSV_PROFILE_STOP");if(!p)continue;FILE*f=fopen(p,"r");if(!f)continue;int id=0;int got=fscanf(f,"%d",&id);fclose(f);if(got!=1)continue;if(id<=finished||id!=s.id.load()||!s.record.load())continue;finished=id;s.record=false;
 bool traced=s.trace.exchange(false);if(traced){nvtxRangePushA("profiler.drain_after_response");int count=0;cudaGetDeviceCount(&count);for(int d=0;d<count;++d){cudaSetDevice(d);auto rc=cudaDeviceSynchronize();if(rc!=cudaSuccess){fprintf(stderr,"PROFILE drain failed %d\n",int(rc));abort();}}nvtxRangePop();auto rc=cudaProfilerStop();if(rc!=cudaSuccess)abort();}
 if(auto*a=getenv("DSV_PROFILE_ACK")){if(FILE*o=fopen(a,"w")){fprintf(o,"%d\n",id);fclose(o);}}fprintf(stderr,"PROFILE_FINISHED case=%d trace=%d\n",id,traced);
 }});
}
__attribute__((destructor))static void stop(){auto&s=S();s.quit=true;if(s.watcher.joinable())s.watcher.join();if(s.log)fclose(s.log);}
