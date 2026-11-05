#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <openssl/evp.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
using json=nlohmann::ordered_json;
static void need(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static std::string sha(const void *p,size_t n){unsigned char v[32];unsigned len;need(EVP_Digest(p,n,v,&len,EVP_sha256(),nullptr)==1,"hash");std::ostringstream out;for(unsigned i=0;i<len;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v[i]);return out.str();}
int main(int argc,char **argv){json result={{"complete",false},{"cases",json::array()}};auto save=[&]{if(argc>1)std::ofstream(argv[1])<<result.dump(2)<<'\n';};try{need(argc==2,"output path");ggml_backend_load_all();std::vector<ggml_backend_t> backends{ggml_backend_cuda_init(0),ggml_backend_cuda_init(5)};need(backends[0]&&backends[1],"backends");
for(auto type:{GGML_TYPE_Q3_K,GGML_TYPE_Q4_K})for(auto dims:{std::pair<int,int>{6144,2048},{2048,6144}}){int k=dims.first,m=dims.second,experts=8,used=6,tokens=6;size_t total=size_t(k)*m*experts;std::vector<float> f(total);for(size_t i=0;i<total;++i)f[i]=(int((i*13+i/7)%10007)-5003)/8192.0f;size_t qbytes=ggml_row_size(type,k)*m*experts;std::vector<uint8_t> q(qbytes);need(ggml_quantize_chunk(type,f.data(),q.data(),0,m*experts,k,nullptr)==qbytes,"quantization");f.clear();f.shrink_to_fit();
for(int bi=0;bi<2;++bi){auto *backend=backends[bi];auto *wc=ggml_init({2*1024*1024,nullptr,true});auto *w=ggml_new_tensor_3d(wc,type,k,m,experts);ggml_set_name(w,"expert.weight");auto *wb=ggml_backend_alloc_ctx_tensors(wc,backend);need(wb,"weight buffer");ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_tensor_set(w,q.data(),0,q.size());
for(int channels:{1,6})for(int pattern:{0,1}){auto *c=ggml_init({8*1024*1024,nullptr,true});auto *b=ggml_new_tensor_3d(c,GGML_TYPE_F32,k,channels,tokens);ggml_set_input(b);auto *allids=ggml_new_tensor_2d(c,GGML_TYPE_I32,experts,tokens);ggml_set_input(allids);auto *ids=ggml_view_2d(c,allids,used,tokens,allids->nb[1],0);auto *wide=ggml_mul_mat_id(c,w,b,ids);ggml_set_output(wide);ggml_set_name(wide,"batch6");ggml_tensor *serial=nullptr;
for(int t=0;t<tokens;++t){auto *x=ggml_view_3d(c,b,k,channels,1,b->nb[1],b->nb[2],t*b->nb[2]);auto *id=ggml_view_2d(c,ids,used,1,ids->nb[1],t*ids->nb[1]);auto *y=ggml_mul_mat_id(c,w,x,id);serial=serial?ggml_concat(c,serial,y,2):y;}
ggml_set_output(serial);ggml_set_name(serial,"six_single_queries");auto *g=ggml_new_graph_custom(c,1024,false);ggml_build_forward_expand(g,wide);ggml_build_forward_expand(g,serial);auto *buf=ggml_backend_alloc_ctx_tensors(c,backend);need(buf,"compute buffer");std::vector<int32_t> vi(experts*tokens);for(int t=0;t<tokens;++t)for(int e=0;e<experts;++e)vi[t*experts+e]=pattern?(t*3+e)%experts:e;ggml_backend_tensor_set(allids,vi.data(),0,vi.size()*sizeof(int32_t));std::string previous;
for(int rep=0;rep<2;++rep){std::vector<float> x(ggml_nelements(b));for(size_t i=0;i<x.size();++i)x[i]=(int((i*17+i/11+rep*29)%2003)-1001)/2048.0f;ggml_backend_tensor_set(b,x.data(),0,x.size()*4);need(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");std::vector<float> a(ggml_nelements(wide)),s(ggml_nelements(serial));need(a.size()==s.size(),"output shape");ggml_backend_tensor_get(wide,a.data(),0,a.size()*4);ggml_backend_tensor_get(serial,s.data(),0,s.size()*4);float diff=0;for(size_t i=0;i<a.size();++i){need(std::isfinite(a[i])&&std::isfinite(s[i]),"finite");diff=std::max(diff,std::fabs(a[i]-s[i]));}bool exact=std::memcmp(a.data(),s.data(),a.size()*4)==0;auto h=sha(a.data(),a.size()*4);result["cases"].push_back({{"device",bi?5:0},{"type",ggml_type_name(type)},{"k",k},{"m",m},{"experts",experts},{"used",used},{"tokens",tokens},{"channels",channels},{"id_stride",ids->nb[1]},{"pattern",pattern},{"rep",rep},{"exact",exact},{"max_abs",diff},{"sha256",h}});save();need(exact,"batch6 vs six singles exact");need(previous.empty()||h!=previous,"fresh input changed output");previous=h;std::cout<<"EXACT "<<(bi?5:0)<<' '<<ggml_type_name(type)<<' '<<k<<' '<<m<<' '<<channels<<' '<<pattern<<' '<<rep<<std::endl;}
ggml_backend_buffer_free(buf);ggml_free(c);}
ggml_backend_buffer_free(wb);ggml_free(wc);}}
for(auto *b:backends)ggml_backend_free(b);ggml_quantize_free();result["complete"]=true;save();}catch(const std::exception&e){result["error"]=e.what();save();std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}return 0;}
