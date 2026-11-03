#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <cstdlib>
using json=nlohmann::ordered_json;
static void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
struct Outputs{std::vector<unsigned char> mask,top,pool;};
static Outputs run(ggml_backend_t*b,int n,int nq,int pattern,bool local,int iterations=2){
 auto*in=ggml_init({4*ggml_tensor_overhead(),nullptr,true});auto*x=ggml_new_tensor_4d(in,GGML_TYPE_F32,n,nq,1,1);auto*pin=ggml_new_tensor_4d(in,GGML_TYPE_F32,(n+7)/8,nq,1,1);ggml_set_input(x);ggml_set_input(pin);auto*ib=ggml_backend_alloc_ctx_tensors(in,b[0]);need(ib,"input alloc");
 auto*ctx=ggml_init({2*1024*1024,nullptr,true});auto*g=ggml_new_graph_custom(ctx,128,false);auto*s=ggml_backend_sched_new(b,nullptr,3,128,true,true);std::vector<ggml_tensor*>early,late;ggml_tensor*mask,*top=nullptr,*pool=nullptr;
 if((n+7)/8<=2048){mask=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,n,nq,1,1),0);late.push_back(mask);}else{
  auto*p=ggml_cont(ctx,x);early.push_back(p);int pad=(n+7)/8*8-n;if(pad){auto*z=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,pad,nq,1,1),-INFINITY);early.push_back(z);p=ggml_concat(ctx,p,z,0);early.push_back(p);}pool=ggml_pool_2d(ctx,p,GGML_OP_POOL_MAX,8,1,8,1,0,0);early.push_back(pool);auto*ps=ggml_add(ctx,pool,pin);late.push_back(ps);top=ggml_cont(ctx,ggml_top_k(ctx,ps,2048));late.push_back(top->src[0]);late.push_back(top);
  auto*keep=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,(n+7)/8,nq,1),-INFINITY);late.push_back(keep);auto*t3=ggml_view_4d(ctx,top,top->ne[0],top->ne[1],top->ne[3],1,top->nb[1],top->nb[2],top->ne[3]*top->nb[3],0);auto*zero=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,t3->ne[0],t3->ne[1],t3->ne[2]),0);late.push_back(zero);keep=ggml_set_rows(ctx,keep,zero,t3);late.push_back(keep);keep=ggml_view_4d(ctx,keep,keep->ne[1],keep->ne[2],1,keep->ne[3],keep->nb[2],keep->nb[3],keep->nb[3],0);keep=ggml_reshape_4d(ctx,keep,1,(n+7)/8,keep->ne[1],1);keep=ggml_repeat_4d(ctx,keep,8,(n+7)/8,nq,1);late.push_back(keep);mask=ggml_reshape_4d(ctx,keep,(n+7)/8*8,nq,1,1);if(pad)mask=ggml_view_4d(ctx,mask,n,nq,1,1,mask->nb[1],mask->nb[2],mask->nb[3],0);
 }
 auto*out=ggml_cont(ctx,mask);late.push_back(out);ggml_set_output(out);if(top)ggml_set_output(top);if(pool)ggml_set_output(pool);ggml_build_forward_expand(g,out);if(top)ggml_build_forward_expand(g,top);if(pool)ggml_build_forward_expand(g,pool);
 for(auto*t:early)ggml_backend_sched_set_tensor_backend(s,t,b[local?0:1]);for(auto*t:late)ggml_backend_sched_set_tensor_backend(s,t,b[1]);need(ggml_backend_sched_alloc_graph(s,g),"explicit allocate");need(ggml_backend_sched_get_tensor_backend(s,out)==b[1],"selector placement");if(pool)need(ggml_backend_sched_get_tensor_backend(s,pool)==b[local?0:1],"pool placement");Outputs z;
 for(int iter=0;iter<iterations;++iter){std::vector<float>v((size_t)n*nq),pi((size_t)(n+7)/8*nq,0);for(int q=0;q<nq;++q){int visible=q==0?0:n-(q+iter)%17;for(int j=0;j<n;++j){float f=float((j*29+q*17+iter*7)%503-251)/13;if(pattern==1)f=-INFINITY;if(pattern==2)f=0;if(pattern==3){if(j%53==0)f=std::numeric_limits<float>::quiet_NaN();else if(j%29==0)f=INFINITY;}v[(size_t)q*n+j]=j>=visible?-INFINITY:f;}if(visible)pi[(size_t)q*((n+7)/8)+(visible-1)/8]=INFINITY;}
  ggml_backend_tensor_set(x,v.data(),0,v.size()*4);ggml_backend_tensor_set(pin,pi.data(),0,pi.size()*4);need(ggml_backend_sched_graph_compute(s,g)==GGML_STATUS_SUCCESS,"compute");auto get=[&](ggml_tensor*t,std::vector<unsigned char>&dst){if(!t)return;size_t old=dst.size();dst.resize(old+ggml_nbytes(t));ggml_backend_tensor_get(t,dst.data()+old,0,ggml_nbytes(t));};get(out,z.mask);get(top,z.top);get(pool,z.pool);
 }
 ggml_backend_sched_free(s);ggml_free(ctx);ggml_backend_buffer_free(ib);ggml_free(in);return z;
}
int main(int argc,char**argv){setenv("GGML_CUDA_GRAPH_STATS","1",1);json j={{"complete",false},{"scope","Existing candidate operators, explicit graph allocation preserves checked backend hints, selector CUDA1, pool/preparation CUDA1 vs CUDA0. Transport/operator check, not production graph_get_cb negative guard or performance test."},{"cases",json::array()}};try{ggml_backend_load_all();ggml_backend_t b[]={ggml_backend_cuda_init(0),ggml_backend_cuda_init(1),ggml_backend_cpu_init()};for(auto*x:b)need(x,"backend");for(int n:{16384,16385,16391,16392})for(int nq:{1,7,511,512,513})for(int pattern=0;pattern<4;++pattern){auto a=run(b,n,nq,pattern,false),c=run(b,n,nq,pattern,true);need(a.mask==c.mask&&a.top==c.top&&a.pool==c.pool,"operator mismatch");j["cases"].push_back({{"rows",n},{"queries",nq},{"pattern",pattern},{"iterations",2},{"mask_bytes",a.mask.size()},{"top_bytes",a.top.size()},{"pool_bytes",a.pool.size()},{"exact",true}});}for(auto*x:b)ggml_backend_free(x);Outputs fixed[2];for(int mode=0;mode<2;++mode){std::cerr<<"FIXED_BEGIN "<<mode<<'\n';ggml_backend_t bb[]={ggml_backend_cuda_init(0),ggml_backend_cuda_init(1),ggml_backend_cpu_init()};fixed[mode]=run(bb,16391,511,0,mode==1,32);for(auto*x:bb)ggml_backend_free(x);std::cerr<<"FIXED_END "<<mode<<'\n';}need(fixed[0].mask==fixed[1].mask&&fixed[0].top==fixed[1].top&&fixed[0].pool==fixed[1].pool,"repeat32 mismatch");j["repeat32"]={{"rows",16391},{"queries",511},{"iterations",32},{"exact",true}};j["actual_placement_checked"]=true;j["complete"]=true;}catch(const std::exception&e){j["error"]=e.what();}std::ofstream(argv[1])<<j.dump(2)<<'\n';std::cout<<j["complete"]<<' '<<j["cases"].size()<<'\n';return j["complete"]?0:1;}
