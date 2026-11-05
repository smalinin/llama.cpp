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
#include "counts.h"
#include <chrono>
#include <algorithm>
#include <dlfcn.h>
int main(int argc,char**argv){json out={{"complete",false},{"cases",json::array()},{"protocol","Fixed-prefix target all-logits calls, restore/sync outside timer. No proposal selection or measured feature bridge. Raw-logit output transport differs from server backend sampling; not native end-to-end TG."}};std::filesystem::path dest=argv[1];std::filesystem::create_directories(dest);auto save=[&]{std::ofstream(dest/"results.json")<<out.dump(2)<<'\n';};try{std::ifstream f(argv[2]);auto input=json::parse(f);common_init();common_params p;std::vector<char*>args{argv[0]};for(int i=3;i<argc;++i)args.push_back(argv[i]);need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"params");p.fit_params=false;p.warmup=false;p.n_outputs_max=p.n_outputs_max_per_seq=6;p.speculative.draft.backend_sampling=false;p.speculative.draft.n_max=5;setenv("LLAMA_DSPARK_CUM_P_MIN","0",1);setenv("LLAMA_DSPARK_EMPTY_COOLDOWN","0",1);unsetenv("LLAMA_DSPARK_GPU_FEATURES");unsetenv("LLAMA_DSPARK_GPU_FEATURES_LOCAL");unsetenv("LLAMA_DSPARK_GPU_FEATURES_ASYNC");unsetenv("LLAMA_DSV41_PREFILL_SKIP");llama_backend_init();auto init=common_init_from_params(p);auto*t=init->context();need(t,"target");common_threadpools tp;tp.init(t,p);auto pd=common_base_params_to_speculative(p);auto di=common_speculative_init_from_params(pd,init->model(),t);auto*d=di->context();need(d,"draft");common_threadpools dp;dp.init(d,pd);p.speculative.draft.ctx_tgt=t;p.speculative.draft.ctx_dft=d;common_speculative_ptr spec(common_speculative_init(p.speculative,1));need(bool(spec),"spec");need(llama_set_dflash_prefill_tail(t,d),"tail registration");need(llama_n_rs_seq(t)==5,"five rollback snapshots");need(t->get_cparams().pipeline_parallel,"pipeline enabled");out["n_rs_seq"]=llama_n_rs_seq(t);std::cerr<<"READY\n";int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(t)));


using Begin=void(*)();using End=void(*)(uint64_t*);auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"dsv_width_count_begin"));auto end=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"dsv_width_count_end"));need(bool(begin)==bool(end),"counter pair");out["instrumented"]=bool(begin);auto tok=input["tokens"].get<Tokens>();need(tok.size()>90118,"input length");
for(int prefix:{8192,90112}){
 llama_memory_clear(llama_get_memory(t),true);llama_memory_clear(llama_get_memory(d),true);common_speculative_set_state(spec.get(),0,{});
 for(int pos=0;pos<prefix;){int n=std::min(8192,prefix-pos);apply(t,d,spec.get(),part(tok,pos,n),pos,false,false);pos+=n;std::cerr<<"PREFILL "<<prefix<<' '<<pos<<'\n';}
 auto base=state(t);auto basehash=sha(base);std::map<int,std::string> ref,states;Floats one;json cases=json::array();
 for(int pass=0;pass<2;++pass)for(int i=0;i<6;++i){int n=pass?6-i:i+1;auto b=llama_batch_init(n,0,1);b.n_tokens=n;for(int k=0;k<n;++k){b.token[k]=tok[prefix+k];b.pos[k]=prefix+k;b.n_seq_id[k]=1;b.seq_id[k][0]=0;b.logits[k]=true;}
  int warm=pass?1:2;
  for(int rep=-warm;rep<5;++rep){restore(t,base);llama_synchronize(t);uint64_t counts[counter_count]{};auto perf=llama_perf_context(t);if(begin)begin();auto before=std::chrono::steady_clock::now();int rc=llama_decode(t,b);llama_synchronize(t);auto after=std::chrono::steady_clock::now();if(end)end(counts);need(rc==0,"verification decode");auto v=logits(t,n);auto h=sha(v);if(!ref.count(n))ref[n]=h;else need(ref[n]==h,"same-width repeat logits");if(n==1&&one.empty())one=v;float delta=0;for(int k=0;k<nv;++k)delta=std::max(delta,std::fabs(v[k]-one[k]));json row={{"pass",pass},{"queries",n},{"rep",rep},{"kind",rep<0?"warmup":"measured"},{"us",std::chrono::duration<double,std::micro>(after-before).count()},{"logits_sha256",h},{"first_row_max_abs_vs_width1",delta},{"first_row_argmax",int(std::max_element(v.begin(),v.begin()+nv)-v.begin())},{"splits",ggml_backend_sched_get_n_splits(t->get_sched())},{"graphs_reused_delta",llama_perf_context(t).n_reused-perf.n_reused}};if(begin){json c;for(int k=0;k<counter_count;++k)c[counter_names[k]]=counts[k];row["counts"]=c;}
   if(rep==4){auto st=sha(state(t));row["target_state_sha256"]=st;if(!states.count(n))states[n]=st;else need(states[n]==st,"same-width state replay");}
   row["prefix"]=prefix;out["observations"].push_back(row);cases.push_back(row);out["progress"]={{"prefix",prefix},{"pass",pass},{"queries",n},{"rep",rep}};save();
  }llama_batch_free(b);std::cerr<<"WIDTH "<<prefix<<' '<<pass<<' '<<n<<'\n';
 }
 restore(t,base);need(sha(state(t))==basehash,"base restore exact");out["cases"].push_back({{"prefix",prefix},{"base_state_sha256",basehash},{"base_state_bytes",base.size()},{"fixed_proposal_ids",part(tok,prefix,6)},{"rows",cases},{"same_width_logits_state_exact",true},{"base_restore_exact",true}});save();
}
out["complete"]=true;save();std::cerr<<"WIDTH_SCREEN_COMPLETE\n";
}catch(const std::exception&e){out["error"]=e.what();save();std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}return 0;}
