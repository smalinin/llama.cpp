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
int main(int argc,char **argv) {
    json result={{"complete",false},{"cases",json::array()}};
    auto save=[&]{if(argc==2)std::ofstream(argv[1])<<result.dump(2)<<'\n';};
    try {
        need(argc==2,"output path");
        std::vector<ggml_backend_t> backends{ggml_backend_cuda_init(0),ggml_backend_cuda_init(5)};
        need(backends[0]&&backends[1],"backends");
        for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
            for(auto dims:{std::pair<int,int>{6144,384},{6144,512},{1536,8192},{512,128},{128,64}}) {
                int k=dims.first,m=dims.second;
                std::vector<float> wf(size_t(k)*m);
                for(size_t i=0;i<wf.size();++i)wf[i]=(int((i*13+i/7)%10007)-5003)/8192.0f;
                std::vector<uint8_t> weights(wf.size()*ggml_type_size(type));
                if(type==GGML_TYPE_F32)std::memcpy(weights.data(),wf.data(),weights.size());
                if(type==GGML_TYPE_F16)ggml_fp32_to_fp16_row(wf.data(),(ggml_fp16_t*)weights.data(),wf.size());
                if(type==GGML_TYPE_BF16)ggml_fp32_to_bf16_row(wf.data(),(ggml_bf16_t*)weights.data(),wf.size());
                for(int bi=0;bi<2;++bi) {
                    auto *backend=backends[bi];
                    auto *wc=ggml_init({2*1024*1024,nullptr,true});
                    auto *w=ggml_new_tensor_2d(wc,type,k,m);
                    auto *wb=ggml_backend_alloc_ctx_tensors(wc,backend);
                    need(wb,"weight buffer");
                    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_tensor_set(w,weights.data(),0,weights.size());
                    for(int channels:{1,2,64})for(int pad:{0,2}) {
                        const int tokens=6;
                        auto *c=ggml_init({8*1024*1024,nullptr,true});
                        auto *store=ggml_new_tensor_3d(c,GGML_TYPE_F32,k+pad,tokens+1,channels);
                        ggml_set_input(store);
                        auto *cur=ggml_view_3d(c,store,k,tokens,channels,store->nb[1],store->nb[2],0);
                        auto *x=ggml_view_4d(c,cur,k,1,channels,tokens,cur->nb[1],cur->nb[2],cur->nb[1],0);
                        auto *y=ggml_mul_mat(c,w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
                        auto *wide=ggml_cont(c,ggml_permute(c,y,0,3,2,1));
                        ggml_set_output(wide);ggml_set_name(wide,"six_sample_grid");
                        ggml_tensor *serial=nullptr;
                        for(int t=0;t<tokens;++t) {
                            auto *a=ggml_view_3d(c,cur,k,1,channels,cur->nb[1],cur->nb[2],t*cur->nb[1]);
                            auto *b=ggml_mul_mat(c,w,a);ggml_mul_mat_set_prec(b,GGML_PREC_F32);
                            serial=serial?ggml_concat(c,serial,b,1):b;
                        }
                        ggml_set_output(serial);ggml_set_name(serial,"six_single_columns");
                        auto *g=ggml_new_graph_custom(c,1024,false);
                        ggml_build_forward_expand(g,wide);ggml_build_forward_expand(g,serial);
                        need(ggml_backend_supports_op(backend,y),"sample op support");
                        auto *buf=ggml_backend_alloc_ctx_tensors(c,backend);need(buf,"compute buffer");
                        std::string previous;
                        for(int rep=0;rep<2;++rep) {
                            std::vector<float> v(ggml_nelements(store));
                            for(size_t i=0;i<v.size();++i)v[i]=(int((i*17+i/11+rep*29)%2003)-1001)/2048.0f;
                            ggml_backend_tensor_set(store,v.data(),0,v.size()*sizeof(float));
                            need(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");
                            std::vector<float>a(ggml_nelements(wide)),b(ggml_nelements(serial));
                            need(a.size()==b.size(),"output shape");
                            ggml_backend_tensor_get(wide,a.data(),0,a.size()*sizeof(float));
                            ggml_backend_tensor_get(serial,b.data(),0,b.size()*sizeof(float));
                            float diff=0;size_t changed=0;
                            for(size_t i=0;i<a.size();++i) {
                                need(std::isfinite(a[i])&&std::isfinite(b[i]),"finite");
                                diff=std::max(diff,std::fabs(a[i]-b[i]));
                                changed+=std::memcmp(&a[i],&b[i],sizeof(float))!=0;
                            }
                            bool exact=changed==0;auto h=sha(a.data(),a.size()*sizeof(float));
                            result["cases"].push_back({{"device",bi?5:0},{"type",ggml_type_name(type)},{"k",k},{"m",m},{"channels",channels},{"pad",pad},{"rep",rep},{"exact",exact},{"changed",changed},{"max_abs",diff},{"sha256",h},{"single_sha256",sha(b.data(),b.size()*sizeof(float))}});
                            save();need(exact,"sample6 vs six singles exact");need(previous.empty()||h!=previous,"fresh input changed output");previous=h;
                        }
                        ggml_backend_buffer_free(buf);ggml_free(c);
                    }
                    ggml_backend_buffer_free(wb);ggml_free(wc);
                }
            }
        }
        for(auto *b:backends)ggml_backend_free(b);
        result["complete"]=true;save();std::cout<<"FLOAT6_EXACT "<<result["cases"].size()<<std::endl;
    }catch(const std::exception&e){result["error"]=e.what();save();std::cerr<<e.what()<<std::endl;return 1;}
    return 0;
}
