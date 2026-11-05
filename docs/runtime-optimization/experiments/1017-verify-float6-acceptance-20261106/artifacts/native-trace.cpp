#include "llama.h"
#include "nlohmann/json.hpp"
#include <openssl/evp.h>
#include <dlfcn.h>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <iomanip>
using json=nlohmann::ordered_json;
namespace {
struct Frame { bool target=false;int epoch=0,marker=0;uint64_t decode=0;std::vector<int32_t> outputs,positions; };
std::mutex lock;
std::map<llama_context*,Frame> frames;
uint64_t serial=0,reads=0;
template<class F>F next(const char *name){auto*p=dlsym(RTLD_NEXT,name);if(!p){std::fprintf(stderr,"NATIVE_TRACE missing %s\n",name);std::abort();}return reinterpret_cast<F>(p);}
std::filesystem::path root(){const char*p=std::getenv("DSV41_NATIVE_TRACE_DIR");if(!p)std::abort();return p;}
void emit(const json&j){static std::ofstream f(root()/"events.jsonl");f<<j.dump()<<'\n';f.flush();if(!f){std::fprintf(stderr,"NATIVE_TRACE write failed\n");std::abort();}}
std::string sha(const void*p,size_t n){unsigned char v[32];unsigned len;if(EVP_Digest(p,n,v,&len,EVP_sha256(),nullptr)!=1)std::abort();std::ostringstream s;for(unsigned i=0;i<len;++i)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v[i]);return s.str();}
uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,sizeof(u));return u;}
}
extern "C" int32_t llama_decode(llama_context *ctx,llama_batch batch){
    static auto fn=next<decltype(&llama_decode)>("llama_decode");
    const int layers=llama_model_n_layer(llama_get_model(ctx));
    {
        std::lock_guard<std::mutex> guard(lock);auto &s=frames[ctx];s.target=layers>10;s.decode=++serial;s.outputs.clear();s.positions.clear();
        if(s.target&&batch.pos&&batch.n_tokens&&batch.pos[0]==0){++s.epoch;s.marker=batch.n_tokens;}
        json row={{"kind","decode"},{"id",s.decode},{"target",s.target},{"layers",layers},{"epoch",s.epoch},{"request_first_batch",s.marker},{"n_tokens",batch.n_tokens},{"tokens",json::array()},{"positions",json::array()},{"outputs",json::array()}};
        for(int i=0;i<batch.n_tokens;++i){if(batch.token)row["tokens"].push_back(batch.token[i]);if(batch.pos){row["positions"].push_back(batch.pos[i]);s.positions.push_back(batch.pos[i]);}if(batch.pos&&(batch.logits?batch.logits[i]!=0:i==batch.n_tokens-1))s.outputs.push_back(batch.pos[i]);}
        row["outputs"]=s.outputs;emit(row);
    }
    return fn(ctx,batch);
}
extern "C" float * llama_get_logits_ith(llama_context *ctx,int32_t idx){
    static auto fn=next<decltype(&llama_get_logits_ith)>("llama_get_logits_ith");
    float *values=fn(ctx,idx);
    std::lock_guard<std::mutex> guard(lock);auto it=frames.find(ctx);if(!values||it==frames.end()||!it->second.target)return values;
    const auto&s=it->second;const auto &positions=idx<0?s.outputs:s.positions;const int i=idx<0?int(positions.size())+idx:idx;
    if(s.marker!=7676||i<0||size_t(i)>=positions.size())return values;
    const int pred=positions[i]+1;if(pred<8192||pred>8213)return values;
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));std::vector<int> top;size_t bad=0;
    for(int t=0;t<nv;++t){if(!std::isfinite(values[t])){++bad;continue;}auto p=top.begin();while(p!=top.end()&&(values[*p]>values[t]||(values[*p]==values[t]&&*p<t)))++p;if(size_t(p-top.begin())<10){top.insert(p,t);if(top.size()>10)top.pop_back();}}
    json ranking=json::array();for(int t:top)ranking.push_back({{"id",t},{"value",values[t]},{"bits",bits(values[t])}});
    const std::string file="logits-"+std::to_string(++reads)+"-pos"+std::to_string(pred)+".f32";
    {std::ofstream f(root()/file,std::ios::binary);f.write(reinterpret_cast<const char*>(values),size_t(nv)*sizeof(float));if(!f)std::abort();}
    emit({{"kind","logits"},{"decode",s.decode},{"epoch",s.epoch},{"request_first_batch",s.marker},{"idx",idx},{"prediction_pos",pred},{"vocab",nv},{"raw",file},{"sha256",sha(values,size_t(nv)*sizeof(float))},{"nonfinite",bad},{"top",ranking},{"logit469",values[469]},{"logit27344",values[27344]}});
    return values;
}
