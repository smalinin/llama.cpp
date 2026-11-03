#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <fstream>
#include <iostream>
#include <stdexcept>
using json=nlohmann::ordered_json;
static void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
struct Out {std::vector<unsigned char> masks, tops, blocks;};
static Out run(int n,int nq,int pattern,bool local,int iters=2) {
 ggml_backend_t b[]={ggml_backend_cuda_init(0),ggml_backend_cuda_init(1),ggml_backend_cpu_init()};for(auto*p:b)need(p,"backend");
 auto*in=ggml_init({4*ggml_tensor_overhead(),nullptr,true});auto*x=ggml_new_tensor_4d(in,GGML_TYPE_F32,n,nq,1,1);auto*pi=ggml_new_tensor_4d(in,GGML_TYPE_F32,(n+7)/8,nq,1,1);ggml_set_input(x);ggml_set_input(pi);auto*ib=ggml_backend_alloc_ctx_tensors(in,b[0]);need(ib,"input");
 auto*ctx=ggml_init({4*1024*1024,nullptr,true});auto*g=ggml_new_graph_custom(ctx,256,false);auto*s=ggml_backend_sched_new(b,nullptr,3,256,true,true);
 std::vector<ggml_tensor*>src;int nb=(n+7)/8,np=nb*8,pad=np-n;auto*p=ggml_cont(ctx,x);src.push_back(p);ggml_tensor*keep=nullptr;
 if(nb>2048) {
  if(pad){auto*z=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,pad,nq,1,1),-INFINITY);src.push_back(z);p=ggml_concat(ctx,p,z,0);src.push_back(p);}
  auto*pool=ggml_pool_2d(ctx,p,GGML_OP_POOL_MAX,8,1,8,1,0,0);src.push_back(pool);auto*ps=ggml_add(ctx,pool,pi);src.push_back(ps);auto*top=ggml_cont(ctx,ggml_top_k(ctx,ps,2048));src.push_back(top->src[0]);src.push_back(top);
  keep=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,nb,nq,1),-INFINITY);src.push_back(keep);auto*tv=ggml_view_4d(ctx,top,top->ne[0],top->ne[1],top->ne[3],1,top->nb[1],top->nb[2],top->ne[3]*top->nb[3],0);auto*z=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,tv->ne[0],tv->ne[1],tv->ne[2]),0);src.push_back(z);keep=ggml_set_rows(ctx,keep,z,tv);src.push_back(keep);
 } else {keep=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,nb,nq,1),0);src.push_back(keep);}
 ggml_set_name(keep,"operator_block_keep");ggml_set_output(keep);
 auto expand=[&](int dev){auto*r=ggml_repeat_4d(ctx,keep,8,nb,nq,1);ggml_format_name(r,"operator_repeat_%d",dev);ggml_backend_sched_set_tensor_backend(s,r,b[dev]);auto*v=ggml_reshape_4d(ctx,r,np,nq,1,1);if(pad)v=ggml_view_4d(ctx,v,n,nq,1,1,v->nb[1],v->nb[2],v->nb[3],0);return std::make_pair(v,r);};
 auto legacy=expand(0);std::array<ggml_tensor*,2> masks{},tops{},repeats{};
 for(int d=0;d<2;++d){auto v=local?expand(d):legacy;repeats[d]=v.second;auto*mask=ggml_cont(ctx,v.first);ggml_backend_sched_set_tensor_backend(s,mask,b[d]);ggml_set_output(mask);masks[d]=mask;auto*add=ggml_add(ctx,x,v.first);ggml_backend_sched_set_tensor_backend(s,add,b[d]);auto*t=ggml_cont(ctx,ggml_top_k(ctx,add,std::min(n,2048)));ggml_backend_sched_set_tensor_backend(s,t->src[0],b[d]);ggml_backend_sched_set_tensor_backend(s,t,b[d]);ggml_set_output(t);tops[d]=t;ggml_build_forward_expand(g,mask);ggml_build_forward_expand(g,t);}
 ggml_build_forward_expand(g,keep);for(auto*t:src)ggml_backend_sched_set_tensor_backend(s,t,b[0]);need(ggml_backend_sched_alloc_graph(s,g),"alloc");for(int d=0;d<2;++d){need(ggml_backend_sched_get_tensor_backend(s,masks[d])==b[d],"output placement");need(ggml_backend_sched_get_tensor_backend(s,repeats[d])==b[local?d:0],"repeat placement");}
 Out out;auto get=[&](ggml_tensor*t,std::vector<unsigned char>&v){size_t off=v.size();v.resize(off+ggml_nbytes(t));ggml_backend_tensor_get(t,v.data()+off,0,ggml_nbytes(t));};
 for(int iter=0;iter<iters;++iter){std::vector<float>v((size_t)n*nq),pins((size_t)nb*nq,0);for(int q=0;q<nq;++q){int visible=q==0?0:n-(q+iter)%17;for(int j=0;j<n;++j){float f=float((j*29+q*17+iter*7)%503-251)/13;if(pattern==1)f=-INFINITY;if(pattern==2)f=0;if(pattern==3){if(j%53==0)f=std::numeric_limits<float>::quiet_NaN();else if(j%29==0)f=INFINITY;}v[(size_t)q*n+j]=j>=visible?-INFINITY:f;}if(visible)pins[(size_t)q*nb+(visible-1)/8]=INFINITY;}
 ggml_backend_tensor_set(x,v.data(),0,v.size()*4);ggml_backend_tensor_set(pi,pins.data(),0,pins.size()*4);need(ggml_backend_sched_graph_compute(s,g)==GGML_STATUS_SUCCESS,"compute");for(int d=0;d<2;++d){get(masks[d],out.masks);get(tops[d],out.tops);}get(keep,out.blocks);}
 ggml_backend_sched_free(s);ggml_free(ctx);ggml_backend_buffer_free(ib);ggml_free(in);for(auto*p:b)ggml_backend_free(p);return out;
}
int main(int argc,char**argv){setenv("GGML_CUDA_GRAPH_STATS","1",1);json j={{"complete",false},{"scope","External operator proof, existing CUDA kernels with explicit actual backend assertions; two consumers, pipeline copies/capture. Not full production eligibility/state/speed proof."},{"cases",json::array()}};try{ggml_backend_load_all();for(int n:{16384,16385,16391,16392})for(int nq:{1,6,7,511,512,513})for(int p=0;p<4;++p){auto a=run(n,nq,p,false),b=run(n,nq,p,true);need(a.masks==b.masks&&a.tops==b.tops&&a.blocks==b.blocks,"mismatch");j["cases"].push_back({{"rows",n},{"queries",nq},{"pattern",p},{"exact",true}});}for(int n:{98305,172031}){auto a=run(n,7,0,false),b=run(n,7,0,true);need(a.masks==b.masks&&a.tops==b.tops&&a.blocks==b.blocks,"long mismatch");j["cases"].push_back({{"rows",n},{"queries",7},{"exact",true}});}Out a[2];for(int m=0;m<2;++m){std::cerr<<"FIXED_BEGIN "<<m<<'\n';a[m]=run(16391,511,0,m==1,32);std::cerr<<"FIXED_END "<<m<<'\n';}need(a[0].masks==a[1].masks&&a[0].tops==a[1].tops&&a[0].blocks==a[1].blocks,"fixed mismatch");j["repeat32_exact"]=true;j["complete"]=true;}catch(const std::exception&e){j["error"]=e.what();}std::ofstream(argv[1])<<j.dump(2)<<'\n';return j["complete"]?0:1;}
