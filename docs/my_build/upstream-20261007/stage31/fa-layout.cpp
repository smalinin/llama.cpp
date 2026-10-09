#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool b,const char * s){if(!b)throw std::runtime_error(s);}
static std::vector<uint8_t> read(const std::string & p){std::ifstream f(p,std::ios::binary|std::ios::ate);check(bool(f),p.c_str());auto n=f.tellg();f.seekg(0);std::vector<uint8_t> v(n);f.read((char*)v.data(),n);return v;}
#include "ordered-helper.inc"
int main(int argc,char ** argv){
 check(argc==7,"usage: fa-layout input output gpu ordered q8 repeats");std::string in=argv[1],out=argv[2];
 auto qd=read(in+"-src0.bin"),kd=read(in+"-src1.bin"),md=read(in+"-src3.bin"),pd=read(in+"-params.bin"),od=read(in+"-order.bin"),vd=read(in+"-valid.bin");
 const int d=512,heads=64,width=4,nk=kd.size()/(d*2);bool ordered=std::stoi(argv[4]),q8=std::stoi(argv[5]);
 auto backend=ggml_backend_cuda_init(std::stoi(argv[3]));check(backend,"CUDA init");auto ctx=ggml_init({16*1024*1024,nullptr,true});
 auto qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,heads,width);auto q=ggml_permute(ctx,qb,0,2,1,3);
 auto kb=ggml_new_tensor_4d(ctx,q8?GGML_TYPE_Q8_0:GGML_TYPE_F16,d,1,nk,1);auto mb=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,nk,width,1,1);
 auto k=kb;auto m=mb;ggml_tensor * order=nullptr,*valid=nullptr;
 if(ordered){order=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,512,1);valid=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,512,1,1,1);build_attn_ordered(ctx,order,valid,k,m);k=ggml_cast(ctx,k,GGML_TYPE_F16);}
 k=ggml_permute(ctx,k,0,2,1,3);auto y=ggml_flash_attn_ext(ctx,q,k,k,m,1,0,0);std::memcpy(y->op_params,pd.data(),pd.size());
 auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"alloc");
 if(q8){std::vector<float> values(kd.size()/2);ggml_fp16_to_fp32_row((const ggml_fp16_t*)kd.data(),values.data(),values.size());kd.resize(ggml_nbytes(kb));check(ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),kd.data(),0,nk,d,nullptr)==kd.size(),"quantize");}
 ggml_backend_tensor_set(qb,qd.data(),0,qd.size());ggml_backend_tensor_set(kb,kd.data(),0,kd.size());ggml_backend_tensor_set(mb,md.data(),0,md.size());
 if(ordered){ggml_backend_tensor_set(order,od.data(),0,od.size());ggml_backend_tensor_set(valid,vd.data(),0,vd.size());}
 std::vector<float> actual(d*heads*width),again(actual.size());check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
 check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat");ggml_backend_tensor_get(y,again.data(),0,again.size()*4);check(std::memcmp(actual.data(),again.data(),actual.size()*4)==0,"repeat differs");
 for(float f:actual)check(std::isfinite(f),"nonfinite");
 std::ofstream(out+".f32",std::ios::binary).write((char*)actual.data(),actual.size()*4);
 const int reps=std::stoi(argv[6]);auto start=std::chrono::steady_clock::now();for(int i=0;i<reps;++i)check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"bench");ggml_backend_synchronize(backend);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/reps;
 std::printf("{\"nkv\":%d,\"ordered\":%s,\"q8\":%s,\"repeat_equal\":true,\"us\":%.3f}\n",nk,ordered?"true":"false",q8?"true":"false",us);
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
