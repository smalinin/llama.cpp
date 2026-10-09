#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool b,const char * s){if(!b)throw std::runtime_error(s);}
static std::vector<uint8_t> read(const std::string & p){std::ifstream f(p,std::ios::binary|std::ios::ate);check(bool(f),"read");auto n=f.tellg();f.seekg(0);std::vector<uint8_t> v(n);f.read((char*)v.data(),n);return v;}
int main(int argc,char ** argv){
 check(argc==4,"usage: fa-replay input-prefix output-prefix gpu-index");std::string in=argv[1],out=argv[2];
 auto qbdata=read(in+"-src0.bin"),kbdata=read(in+"-src1.bin"),vbdata=read(in+"-src2.bin"),mask=read(in+"-src3.bin"),params=read(in+"-params.bin"),capture=read(in+"-fa.bin");
 const int d=512,heads=64,width=4,nk=kbdata.size()/(d*2);check(qbdata.size()==size_t(d*heads*width*4),"query size");check(mask.size()==size_t(nk*width*2),"mask size");check(vbdata==kbdata,"MLA shared KV");
 auto backend=ggml_backend_cuda_init(std::stoi(argv[3]));check(backend,"CUDA init");auto ctx=ggml_init({16*1024*1024,nullptr,true});
 auto qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,heads,width);auto q=ggml_permute(ctx,qb,0,2,1,3);
 auto k=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,d,nk,1,1);auto m=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,nk,width,1,1);
 auto y=ggml_flash_attn_ext(ctx,q,k,k,m,1,0,0);check(params.size()==sizeof(y->op_params),"params size");std::memcpy(y->op_params,params.data(),params.size());
 auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"alloc");
 ggml_backend_tensor_set(qb,qbdata.data(),0,qbdata.size());ggml_backend_tensor_set(k,kbdata.data(),0,kbdata.size());ggml_backend_tensor_set(m,mask.data(),0,mask.size());
 std::vector<float> actual(d*heads*width),again(actual.size());check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
 check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat");ggml_backend_tensor_get(y,again.data(),0,again.size()*4);check(actual==again,"repeat differs");
 std::ofstream(out+".f32",std::ios::binary).write((char*)actual.data(),actual.size()*4);
 const float * c=(const float*)capture.data();double maximum=0;for(size_t i=0;i<actual.size();++i)maximum=std::max(maximum,std::abs(double(actual[i])-c[i]));
 std::printf("{\"nk\":%d,\"capture_equal\":%s,\"max_abs_vs_capture\":%.12g,\"repeat_equal\":true}\n",nk,std::memcmp(actual.data(),capture.data(),capture.size())==0?"true":"false",maximum);
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
