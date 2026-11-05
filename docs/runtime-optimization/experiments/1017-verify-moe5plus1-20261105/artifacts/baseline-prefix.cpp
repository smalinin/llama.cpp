#include "arg.h"
#include "common.h"
#include "speculative.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include <openssl/evp.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
using json=nlohmann::ordered_json;using Tokens=std::vector<llama_token>;using Floats=std::vector<float>;using Bytes=std::vector<uint8_t>;
static void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static std::string sha(const void*p,size_t n){unsigned char d[32];unsigned l;need(EVP_Digest(p,n,d,&l,EVP_sha256(),nullptr)==1,"digest");std::ostringstream o;for(unsigned i=0;i<l;++i)o<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(d[i]);return o.str();}
template<class T>static std::string sha(const std::vector<T>&v){return sha(v.data(),v.size()*sizeof(T));}
static Bytes state(llama_context*c,bool full=false){llama_synchronize(c);Bytes v(full?llama_state_get_size(c):llama_state_seq_get_size(c,0));need((full?llama_state_get_data(c,v.data(),v.size()):llama_state_seq_get_data(c,v.data(),v.size(),0))==v.size(),"save");return v;}
static void restore(llama_context*c,const Bytes&v){need(llama_state_seq_set_data(c,v.data(),v.size(),0)==v.size(),"restore");}
static Tokens part(const Tokens&t,int p,int n){return{t.begin()+p,t.begin()+p+n};}
static Floats logits(llama_context*c,int n=1){int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c)));Floats v;for(int i=0;i<n;++i){auto*q=llama_get_logits_ith(c,n==1?-1:i);need(q,"logits");for(int j=0;j<nv;++j)need(std::isfinite(q[j]),"nonfinite");v.insert(v.end(),q,q+nv);}return v;}
static Floats apply(llama_context*t,llama_context*d,common_speculative*s,const Tokens&ids,int pos,bool all=false,bool out=true){auto b=llama_batch_init(ids.size(),0,1);b.n_tokens=ids.size();for(int i=0;i<b.n_tokens;++i){b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=all||(out&&i==b.n_tokens-1);}need(llama_decode(t,b)==0,"target decode");Floats v;if(out)v=logits(t,all?ids.size():1);need(common_speculative_process(s,b),"feature process");llama_synchronize(d);llama_batch_free(b);return v;}
static json draft(common_speculative*s,llama_context*d,Tokens&prompt,int pos,int anchor,int k){Tokens ids;Floats conf;auto&p=common_speculative_get_draft_params(s,0);p.drafting=true;p.n_max=k;p.n_past=pos;p.id_last=anchor;p.prompt=&prompt;p.result=&ids;p.confidence=&conf;common_speculative_draft(s);need(ids.size()==size_t(k)&&conf.size()==size_t(k),"draft width");auto v=logits(d,k);return{{"ids",ids},{"confidence",conf},{"logits",sha(v)},{"seq",sha(state(d))},{"global",sha(state(d,true))}};}
static json taps(llama_context * t) {
    uint32_t begin=0,end=0;
    need(llama_get_embeddings_layer_inp_range(t,&begin,&end) && end>begin,"tap range");
    json out={{"begin",begin},{"end",end},{"values",json::array()}};
    for(int lid:{37,38,39}) {
        auto * q=t->get_layer_input_span(lid,begin,end-begin);
        need(q,"tap data");
        out["values"].push_back(sha(q,size_t(end-begin)*llama_model_n_embd(llama_get_model(t))*sizeof(float)));
    }
    return out;
}
static Bytes partial(llama_context*c){llama_synchronize(c);auto fl=LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;Bytes v(llama_state_seq_get_size_ext(c,0,fl));need(llama_state_seq_get_data_ext(c,v.data(),v.size(),0,fl)==v.size(),"partial save");return v;}
static void restore_partial(llama_context*c,const Bytes&v){need(llama_state_seq_set_data_ext(c,v.data(),v.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)==v.size(),"partial restore");}
static void restore_full(llama_context*c,const Bytes&v){need(llama_state_set_data(c,v.data(),v.size())==v.size(),"full restore");}
static bool fault_apply(llama_context*t,common_speculative*s,const Tokens&ids){auto b=llama_batch_init(ids.size(),0,1);b.n_tokens=ids.size();for(int i=0;i<b.n_tokens;++i){b.token[i]=ids[i];b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=false;}need(llama_decode(t,b)==0,"fault target decode");bool ok=common_speculative_process(s,b);llama_batch_free(b);return ok;}
int main(int argc,char**argv){json out={{"complete",false},{"cases",json::array()},{"faults",json::array()}};std::filesystem::path dest=argv[1];std::filesystem::create_directories(dest);auto save=[&]{std::ofstream(dest/"results.json")<<out.dump(2)<<'\n';};try{std::ifstream f(argv[2]);auto input=json::parse(f);input["cases"].erase(input["cases"].begin()+1,input["cases"].end());common_init();common_params p;std::vector<char*>args{argv[0]};for(int i=3;i<argc;++i)args.push_back(argv[i]);need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"params");p.fit_params=false;p.warmup=false;p.n_outputs_max=p.n_outputs_max_per_seq=6;p.speculative.draft.backend_sampling=false;p.speculative.draft.p_min=0;p.speculative.draft.n_max=5;setenv("LLAMA_DSPARK_CUM_P_MIN","0",1);setenv("LLAMA_DSPARK_EMPTY_COOLDOWN","0",1);unsetenv("LLAMA_DSPARK_GPU_FEATURES");unsetenv("LLAMA_DSPARK_GPU_FEATURES_LOCAL");unsetenv("LLAMA_DSPARK_GPU_FEATURES_ASYNC");unsetenv("LLAMA_DSV41_PREFILL_SKIP");llama_backend_init();auto init=common_init_from_params(p);auto*t=init->context();need(t,"target");common_threadpools tp;tp.init(t,p);auto pd=common_base_params_to_speculative(p);auto di=common_speculative_init_from_params(pd,init->model(),t);auto*d=di->context();need(d,"draft");common_threadpools dp;dp.init(d,pd);p.speculative.draft.ctx_tgt=t;p.speculative.draft.ctx_dft=d;common_speculative_ptr spec(common_speculative_init(p.speculative,1));need(bool(spec),"spec");need(llama_set_dflash_prefill_tail(t,d),"tail registration");std::cerr<<"READY\n";int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(t)));

for(auto&corpus:input["cases"]){std::string id=corpus["id"];auto tok=corpus["tokens"].get<Tokens>();std::map<std::string,json>refs;
for(int mode:{0,1,2}){llama_memory_clear(llama_get_memory(t),true);llama_memory_clear(llama_get_memory(d),true);common_speculative_set_state(spec.get(),0,{});const auto empty_t=sha(state(t)),empty_d=sha(state(d));
auto record=[&](std::string key,json row){if(mode==0)refs[key]=row;else if(refs.at(key)!=row){out["mismatch"]={{"corpus",id},{"mode",mode},{"key",key},{"reference",refs.at(key)},{"actual",row}};save();throw std::runtime_error("long mode mismatch");}out["cases"].push_back({{"corpus",id},{"mode",mode},{"key",key},{"result",row}});save();std::cerr<<"EXACT "<<id<<' '<<mode<<' '<<key<<'\n';};
json first;Bytes at90,ad90;for(int pos=0;pos<8192;pos+=8192){apply(t,d,spec.get(),part(tok,pos,8192),pos,false,false);if(pos==0)first={{"taps",taps(t)},{"target_seq",sha(state(t))},{"target_global",sha(state(t,true))},{"draft_seq",sha(state(d))},{"draft_global",sha(state(d,true))}};out["progress"]={{"corpus",id},{"mode",mode},{"prefix",pos+8192}};save();std::cerr<<"PREFILL "<<id<<' '<<mode<<' '<<pos+8192<<'\n';}record("first8K",first);}}out["complete"]=true;out.erase("progress");save();std::cerr<<"BASELINE_PREFIX_DONE\n";return 0;}catch(const std::exception&e){out["error"]=e.what();save();std::cerr<<"PROBE_FAIL "<<e.what()<<'\n';return 1;}}
