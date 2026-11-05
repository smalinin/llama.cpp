#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <algorithm>
using json=nlohmann::ordered_json;
static void check(bool ok,const char*msg){if(!ok)throw std::runtime_error(msg);}
static ggml_status(*cpu_compute)(ggml_backend_t,ggml_cgraph*),(*gpu_compute)(ggml_backend_t,ggml_cgraph*),(*last_compute)(ggml_backend_t,ggml_cgraph*);
static ggml_backend_event_t work_event;
static bool work_recorded=false,fail_once=false;
static int early_count=0,cpu_calls=0;
static ggml_status on_cpu(ggml_backend_t b,ggml_cgraph*g){
 bool lookup=false;for(int i=0;i<ggml_graph_n_nodes(g);++i)lookup|=ggml_graph_node(g,i)->op==GGML_OP_GET_ROWS;
 if(lookup&&work_recorded){auto rc=cudaEventQuery((cudaEvent_t)work_event->context);check(rc==cudaSuccess||rc==cudaErrorNotReady,"query event");++cpu_calls;early_count+=rc==cudaErrorNotReady;}
 return cpu_compute(b,g);
}
static ggml_status on_gpu(ggml_backend_t b,ggml_cgraph*g){auto rc=gpu_compute(b,g);ggml_backend_event_record(work_event,b);work_recorded=true;return rc;}
static ggml_status on_last(ggml_backend_t b,ggml_cgraph*g){if(fail_once){fail_once=false;return GGML_STATUS_FAILED;}return last_compute(b,g);}
static bool eval(ggml_tensor*,bool,void*){return true;}
struct Result{std::vector<std::vector<float>> out;int early=0,calls=0,failures=0;};
static Result run(ggml_backend_t a,ggml_backend_t b,ggml_backend_t cpu,const std::string&variant,int mode){
 if(mode<0)unsetenv("GGML_SCHED_CPU_LOOKUP_EARLY");else setenv("GGML_SCHED_CPU_LOOKUP_EARLY",mode?"1":"0",1);
 const bool parallel=variant!="single_copy";ggml_backend_t bs[]={a,b,cpu};ggml_backend_buffer_type_t bt[]={ggml_backend_get_default_buffer_type(a),ggml_backend_get_default_buffer_type(b),ggml_backend_dev_host_buffer_type(ggml_backend_get_device(a))};
 auto*s=ggml_backend_sched_new(bs,bt,3,16384,parallel,true);check(s,"scheduler");if(parallel){bool active=ggml_backend_sched_set_inputs_async(s,true);check(active!=(variant=="shared_type"||variant=="shared_without_edge"),"async compatibility");}if(variant=="callback")ggml_backend_sched_set_eval_callback(s,eval,nullptr);
 const int d=1024,rows=2048;auto*wc=ggml_init({1024*1024,nullptr,true});auto*w=ggml_new_tensor_2d(wc,GGML_TYPE_F32,d,d);ggml_set_name(w,"identity.weight");auto*wb=ggml_backend_alloc_ctx_tensors(wc,a);check(wb,"GPU weights");ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);std::vector<float>wv(d*d,0);for(int i=0;i<d;++i)wv[i*d+i]=1;ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4);
 auto*tc=ggml_init({1024*1024,nullptr,true});auto*table=ggml_new_tensor_2d(tc,GGML_TYPE_F32,d,rows);ggml_set_name(table,"lookup.weight");auto*tb=ggml_backend_alloc_ctx_tensors(tc,cpu);check(tb,"CPU weights");ggml_backend_buffer_set_usage(tb,variant=="mutable_weight"?GGML_BACKEND_BUFFER_USAGE_COMPUTE:GGML_BACKEND_BUFFER_USAGE_WEIGHTS);std::vector<float>tv(d*rows);for(int i=0;i<rows;++i)for(int k=0;k<d;++k)tv[i*d+k]=float(i%23)/8+float(k%7)/32;ggml_backend_tensor_set(table,tv.data(),0,tv.size()*4);
 Result r;std::vector<ggml_context*>ctxs;early_count=cpu_calls=0;work_recorded=false;const int steps=variant=="eligible"?48:16;
 for(int step=0;step<steps;++step){
  const int shape[]={1,4,6,7,128,511,512,513};int n=variant=="shapes"?shape[step%8]:(variant=="eligible"?512:((step%3)==0?128:((step%3)==1?511:512)));int old_early=early_count;
  auto*c=ggml_init({8*1024*1024,nullptr,true});ctxs.push_back(c);auto*g=ggml_new_graph_custom(c,16384,false);auto*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,n);ggml_set_input(x);ggml_set_name(x,"input");auto*ids=ggml_new_tensor_1d(c,GGML_TYPE_I32,n);ggml_set_input(ids);ggml_set_name(ids,"ids");auto*work=x;std::vector<ggml_tensor*>ops;
  for(int k=0;k<32;++k){work=ggml_mul_mat(c,w,work);ggml_mul_mat_set_prec(work,GGML_PREC_F32);ops.push_back(work);}
  auto*lookup=ggml_get_rows(c,table,ids);ggml_set_name(lookup,"lookup");if(variant=="exported")ggml_set_output(lookup);auto*payload=variant=="cpu_math"?ggml_scale(c,lookup,2):(variant=="view"?ggml_reshape_2d(c,lookup,d,n):lookup);ggml_tensor*reduced=nullptr;if(variant=="shared_without_edge"){reduced=ggml_sum_rows(c,work);ggml_build_forward_expand(g,reduced);ops.push_back(reduced);}auto*out=variant=="shared_without_edge"?ggml_scale(c,payload,1):ggml_add(c,work,payload);ggml_set_output(out);ggml_build_forward_expand(g,out);
  ggml_backend_sched_reset(s);ggml_backend_sched_set_tensor_backend(s,x,cpu);ggml_backend_sched_set_tensor_backend(s,ids,(variant=="gpu_ids"||(variant=="switch"&&step==4))?a:cpu);for(auto*t:ops)ggml_backend_sched_set_tensor_backend(s,t,a);ggml_backend_sched_set_tensor_backend(s,lookup,cpu);if(payload!=lookup)ggml_backend_sched_set_tensor_backend(s,payload,cpu);ggml_backend_sched_set_tensor_backend(s,out,b);check(ggml_backend_sched_alloc_graph(s,g),"allocate");
  if(variant!="unprepared")ggml_backend_sched_prepare_inputs(s);std::vector<float>xv(d*n);std::vector<int32_t>iv(n);for(int i=0;i<n;++i){iv[i]=(step*13+i*7)%rows;for(int k=0;k<d;++k)xv[i*d+k]=float(step+1)+float(k%13)/64;}ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);ggml_backend_tensor_set(ids,iv.data(),0,iv.size()*4);
  if(variant=="failure"&&step==7)fail_once=true;auto status=ggml_backend_sched_graph_compute_async(s,g);if(variant=="failure"&&step==7){check(status==GGML_STATUS_FAILED&&!fail_once,"injected failure");++r.failures;continue;}check(status==GGML_STATUS_SUCCESS,"compute");if((variant=="switch"&&(step==4||step==5))||(variant=="failure"&&step==8))check(early_count==old_early,"unproven predecessor ordering");
  r.out.emplace_back(d*n);ggml_backend_tensor_get_async(b,out,r.out.back().data(),0,ggml_nbytes(out));
  // Check after the final drain; each host output keeps its own storage.
 }
 ggml_backend_sched_synchronize(s);r.early=early_count;r.calls=cpu_calls;
 int step=0;for(auto&v:r.out){if(variant=="failure"&&step==7)++step;int n=v.size()/d;for(int i=0;i<n;++i)for(int k=0;k<d;++k){float expected=(variant=="shared_without_edge"?0:float(step+1)+float(k%13)/64)+(variant=="cpu_math"?2:1)*tv[((step*13+i*7)%rows)*d+k];check(v[i*d+k]==expected,"reference value");}++step;}
 ggml_backend_sched_free(s);for(auto*c:ctxs)ggml_free(c);ggml_backend_buffer_free(tb);ggml_free(tc);ggml_backend_buffer_free(wb);ggml_free(wc);return r;
}
int main(int argc,char**argv){json j={{"complete",false},{"cases",json::array()}};auto save=[&]{if(argc>1)std::ofstream(argv[1])<<j.dump(2)<<'\n';};try{check(argc==2,"output path");ggml_backend_load_all();auto*a=ggml_backend_cuda_init(0);auto*b=ggml_backend_cuda_init(1);auto*cpu=ggml_backend_cpu_init();check(a&&b&&cpu,"backends");ggml_backend_cpu_set_n_threads(cpu,4);cpu_compute=cpu->iface.graph_compute;gpu_compute=a->iface.graph_compute;last_compute=b->iface.graph_compute;work_event=ggml_backend_event_new(ggml_backend_get_device(a));check(work_event,"event");cpu->iface.graph_compute=on_cpu;a->iface.graph_compute=on_gpu;b->iface.graph_compute=on_last;
 for(const auto&v:std::vector<std::string>{"eligible","view","shapes","switch","mutable_weight","exported","gpu_ids","cpu_math","single_copy","callback","unprepared","failure","shared_type","shared_without_edge"}){std::vector<std::vector<float>>ref;for(int mode:{0,1,-1}){auto*selected=b;if(v=="shared_type"||v=="shared_without_edge"){selected=ggml_backend_cuda_init(0);check(selected,"shared backend");selected->iface.graph_compute=on_last;}auto r=run(a,selected,cpu,v,mode);if(selected!=b){selected->iface.graph_compute=last_compute;ggml_backend_free(selected);}if(mode==0)ref=r.out;else check(r.out==ref,"OFF/ON exact");if(mode==0||!(v=="eligible"||v=="view"||v=="shapes"||v=="switch"||v=="failure"))check(r.early==0,"fallback ordering");else check(r.early>0,"early branch overlap");j["cases"].push_back({{"variant",v},{"mode",mode},{"early",r.early},{"cpu_calls",r.calls},{"outputs",r.out.size()},{"failures",r.failures}});save();std::printf("CASE %s mode=%d early=%d calls=%d\n",v.c_str(),mode,r.early,r.calls);std::fflush(stdout);}}
 cpu->iface.graph_compute=cpu_compute;a->iface.graph_compute=gpu_compute;b->iface.graph_compute=last_compute;ggml_backend_event_free(work_event);ggml_backend_free(cpu);ggml_backend_free(b);ggml_backend_free(a);j["complete"]=true;save();}catch(const std::exception&e){j["error"]=e.what();save();std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}return 0;}
