#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-kv-cache-iswa.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <openssl/evp.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
using Bytes=std::vector<uint8_t>;using json=nlohmann::ordered_json;
static void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static std::string sha(const Bytes&v){unsigned char d[32];unsigned n;need(EVP_Digest(v.data(),v.size(),d,&n,EVP_sha256(),nullptr)==1,"digest");std::ostringstream o;for(unsigned i=0;i<n;++i)o<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(d[i]);return o.str();}
struct Writer:llama_io_write_i{Bytes v;void write(const void*p,size_t n)override{auto*b=(const uint8_t*)p;v.insert(v.end(),b,b+n);}void write_tensor(ggml_tensor*t,size_t off,size_t n)override{size_t at=v.size();v.resize(at+n);ggml_backend_tensor_get(t,v.data()+at,off,n);}size_t n_bytes()override{return v.size();}};
static Bytes state(llama_kv_cache_iswa&kv,int seq){Writer w;kv.state_write(w,seq);return w.v;}
static void apply(llama_kv_cache_iswa&kv,const llama_model&model,int pos,int n,bool payload){auto b=llama_batch_init(n,0,1);b.n_tokens=n;for(int i=0;i<n;++i){b.token[i]=1;b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=false;}llama_batch_allocr alloc(1);need(alloc.init(b,model.vocab,&kv,model.hparams.n_embd,1,false),"alloc init");auto ctx=kv.init_batch(alloc,512,false);need(ctx&&ctx->get_status()==LLAMA_MEMORY_STATUS_SUCCESS,"prepare");do{const auto&ub=ctx->get_ubatch();need(ctx->apply(),"apply");if(payload){auto*swa=kv.get_swa();auto&cells=swa->get_cells(0);for(unsigned il=0;il<model.hparams.n_layer_all;++il){auto*k=swa->get_k_storage(il);need(k&&k->data,"storage");Bytes row(k->nb[1]);for(unsigned j=0;j<cells.size();++j){if(cells.is_empty(j))continue;int p=cells.pos_get(j);if(p<ub.pos[0]||p>=ub.pos[0]+int(ub.n_tokens))continue;for(size_t z=0;z<row.size();++z)row[z]=uint8_t((uint64_t(p)*2654435761ull+il*137+z*17)>>(z%13));ggml_backend_tensor_set(k,row.data(),j*k->nb[1],row.size());}}}}while(ctx->next());llama_batch_free(b);}
int main(int argc,char**argv){try{need(argc==3,"model/output args");llama_backend_init();auto mp=llama_model_default_params();mp.no_alloc=true;mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;std::unique_ptr<llama_model> model(llama_model_load_from_file(argv[1],mp));need(bool(model),"model");auto&hp=model->hparams;need(model->arch==LLM_ARCH_DFLASH&&hp.n_layer_all==3&&hp.n_swa==129,"metadata");hp.no_alloc=false;llama_kv_cache_iswa kv(*model,GGML_TYPE_F16,GGML_TYPE_F16,false,false,false,false,100352,1,512,256,nullptr,{}, {},{});need(kv.get_swa()->get_size()==768&&!kv.get_swa()->has_cell_ext(),"cache contract");json out={{"complete",false},{"capacity",kv.get_swa()->get_size()},{"window",hp.n_swa},{"cases",json::array()}};int neg=0;
for(int origin:{0,37,90112})for(int prefix:{0,1,127,512,513,8192})for(int n:{3584,4096,7676,7681,8191,8192}){int start=origin+prefix;kv.clear(true);if(prefix)apply(kv,*model,origin,prefix,true);apply(kv,*model,start,n,true);auto full=state(kv,-1),seq=state(kv,0);const int begin=(n-768)/512*512;kv.clear(true);if(prefix)apply(kv,*model,origin,prefix,true);if(begin)apply(kv,*model,start,begin,false);apply(kv,*model,start+begin,n-begin,true);need(full==state(kv,-1),"full mismatch");need(seq==state(kv,0),"seq mismatch");
kv.clear(true);if(prefix)apply(kv,*model,origin,prefix,true);int short_begin=(n-512)/512*512;if(short_begin)apply(kv,*model,start,short_begin,false);apply(kv,*model,start+short_begin,n-short_begin,true);bool same_full=full==state(kv,-1),same_seq=seq==state(kv,0);if(n%512==0){need(!same_full&&same_seq,"negative insufficient tail");++neg;}out["cases"].push_back({{"origin",origin},{"prefix",prefix},{"count",n},{"first_valid",begin},{"tail",n-begin},{"full_sha",sha(full)},{"seq_sha",sha(seq)},{"short_full_equal",same_full},{"short_seq_equal",same_seq}});}
out["complete"]=true;out["negative_checks"]=neg;std::ofstream(argv[2])<<out.dump(2)<<'\n';std::cout<<"PASS cases="<<out["cases"].size()<<" negative="<<neg<<'\n';return 0;}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
