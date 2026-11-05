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
#include <cstring>
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

#include "llama-io.h"
#include "llama-kv-cache-dsv4.h"
struct StateDescribe:llama_io_write_i {
 const Bytes &data;size_t offset=8,meta_begin=8;json parts=json::array(),bad=json::array();
 StateDescribe(const Bytes&v):data(v){}
 void flush(){if(offset>meta_begin)parts.push_back({{"kind","metadata"},{"offset",meta_begin},{"bytes",offset-meta_begin},{"sha256",sha(data.data()+meta_begin,offset-meta_begin)}});meta_begin=offset;}
 void write(const void *src,size_t size)override{need(offset+size<=data.size(),"descriptor meta bounds");if(std::memcmp(src,data.data()+offset,size))bad.push_back({{"offset",offset},{"bytes",size}});offset+=size;}
 void write_tensor(ggml_tensor*t,size_t first,size_t size)override{flush();need(offset+size<=data.size(),"descriptor tensor bounds");json row={{"kind","tensor"},{"name",t->name},{"type",ggml_type_name(t->type)},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"tensor_offset",first},{"offset",offset},{"bytes",size},{"sha256",sha(data.data()+offset,size)}};auto *buf=t->buffer;if(!buf&&t->view_src)buf=t->view_src->buffer;if(buf){auto *bt=ggml_backend_buffer_get_type(buf);auto *dev=ggml_backend_buft_get_device(bt);row["owner"]=dev?ggml_backend_dev_name(dev):"none";}parts.push_back(row);offset+=size;meta_begin=offset;}
 size_t n_bytes()override{return offset;}
};
static json position(llama_context*t){auto *m=dynamic_cast<llama_kv_cache_dsv4*>(llama_get_memory(t));need(m,"DSV memory");return{{"max",llama_memory_seq_pos_max(llama_get_memory(t),0)},{"rs_idx",m->get_rs_idx()}};}
static json describe(llama_context*t,const Bytes &data){StateDescribe w(data);llama_get_memory(t)->state_write(w,0,0);w.flush();need(w.n_bytes()==data.size(),"descriptor size");return{{"bytes",data.size()},{"sha256",sha(data)},{"position",position(t)},{"parts",w.parts},{"metadata_changed_while_describing",w.bad}};}
static json byte_diff(const Bytes&a,const Bytes&b){json result={{"size_before",a.size()},{"size_after",b.size()},{"changes",json::array()}};size_t count=0;for(size_t i=0;i<std::min(a.size(),b.size());++i)if(a[i]!=b[i]){++count;if(result["changes"].size()<64){size_t base=i&~size_t(3);json row={{"offset",i},{"before",a[i]},{"after",b[i]}};if(base+4<=std::min(a.size(),b.size())){uint32_t av,bv;float af,bf;std::memcpy(&av,a.data()+base,4);std::memcpy(&bv,b.data()+base,4);std::memcpy(&af,&av,4);std::memcpy(&bf,&bv,4);row["word_offset"]=base;row["bits_before"]=av;row["bits_after"]=bv;if(std::isfinite(af)&&std::isfinite(bf)){row["float_before"]=af;row["float_after"]=bf;}}result["changes"].push_back(row);}}result["changed_bytes"]=count;return result;}

#include "counts.h"
#include <dlfcn.h>
static std::filesystem::path blobs;
static void blob(const Bytes &v) {
    const auto path=blobs/(sha(v)+".bin");
    if (!std::filesystem::exists(path)) {
        std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size());need(bool(f),"blob write");
    }
}
static Bytes read_blob(const std::string &hash) {
    std::ifstream f(blobs/(hash+".bin"),std::ios::binary|std::ios::ate);need(bool(f),"blob open");auto n=f.tellg();need(n>=0,"blob size");Bytes v(size_t(n),0);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),v.size());need(bool(f)&&sha(v)==hash,"blob integrity");return v;
}
static void float_blob(const Floats &v) { Bytes b(v.size()*sizeof(float));if(!v.empty())std::memcpy(b.data(),v.data(),b.size());blob(b); }
int main(int argc,char **argv) {
    json out={{"complete",false},{"cases",json::array()},{"internal_exact",true}};
    std::filesystem::path dest=argv[1], bundle=argv[3];bool generate=std::string(argv[4])=="generate";blobs=bundle/"blobs";
    std::filesystem::create_directories(dest);std::filesystem::create_directories(blobs);
    auto save=[&]{std::ofstream(dest/"results.json")<<out.dump(2)<<'\n';};
    try {
        std::ifstream input_file(argv[2]);auto input=json::parse(input_file);auto tok=input["cases"][0]["tokens"].get<Tokens>();const int prefix=90112;
        need(tok.size()>prefix+6,"input length");common_init();common_params p;std::vector<char*> args{argv[0]};for(int i=5;i<argc;++i)args.push_back(argv[i]);need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"params");
        p.fit_params=false;p.warmup=false;p.n_outputs_max=p.n_outputs_max_per_seq=6;p.speculative.draft.backend_sampling=false;p.speculative.draft.p_min=0;p.speculative.draft.n_max=5;
        setenv("LLAMA_DSPARK_CUM_P_MIN","0",1);setenv("LLAMA_DSPARK_EMPTY_COOLDOWN","0",1);unsetenv("LLAMA_DSPARK_GPU_FEATURES");unsetenv("LLAMA_DSPARK_GPU_FEATURES_LOCAL");unsetenv("LLAMA_DSPARK_GPU_FEATURES_ASYNC");unsetenv("LLAMA_DSV41_PREFILL_SKIP");
        llama_backend_init();auto init=common_init_from_params(p);auto*t=init->context();need(t,"target");common_threadpools tp;tp.init(t,p);auto pd=common_base_params_to_speculative(p);auto di=common_speculative_init_from_params(pd,init->model(),t);auto*d=di->context();need(d,"draft");common_threadpools dp;dp.init(d,pd);
        p.speculative.draft.ctx_tgt=t;p.speculative.draft.ctx_dft=d;common_speculative_ptr spec(common_speculative_init(p.speculative,1));need(bool(spec),"spec");need(llama_set_dflash_prefill_tail(t,d),"tail registration");need(llama_n_rs_seq(t)==5&&t->get_cparams().pipeline_parallel,"fixed context");std::cerr<<"READY\n";
        using Begin=void(*)();using End=void(*)(uint64_t*);auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"dsv_width_count_begin"));auto end=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"dsv_width_count_end"));need(begin&&end,"counter loaded");
        json seed;Tokens prompt=part(tok,0,prefix);Bytes bs,ds,bf,df;int anchor=0;out["generate"]=generate;out["selector"]=std::getenv("LLAMA_DSV41_VERIFY_MOE_WIDE");
        if(generate) {
            need(out["selector"]=="0","seed OFF");llama_memory_clear(llama_get_memory(t),true);llama_memory_clear(llama_get_memory(d),true);common_speculative_set_state(spec.get(),0,{});
            json first;
            for(int pos=0;pos<81920;pos+=8192) { apply(t,d,spec.get(),part(tok,pos,8192),pos,false,false);if(!pos)first={{"taps",taps(t)},{"target_seq",sha(state(t))},{"target_global",sha(state(t,true))},{"draft_seq",sha(state(d))},{"draft_global",sha(state(d,true))}};std::cerr<<"PREFILL "<<pos+8192<<'\n'; }
            apply(t,d,spec.get(),part(tok,81920,7676),81920,false,false);apply(t,d,spec.get(),part(tok,89596,512),89596,false,false);auto last=apply(t,d,spec.get(),part(tok,90108,4),90108);int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(t)));anchor=std::max_element(last.end()-nv,last.end())-(last.end()-nv);
            bs=state(t);ds=state(d);bf=state(t,true);df=state(d,true);common_speculative_begin(spec.get(),0,prompt);Bytes sp;common_speculative_get_state(spec.get(),0,sp);need(sp.empty(),"stateless spec");
            seed={{"complete",true},{"prefix",prefix},{"prompt_ids_sha256",sha(prompt)},{"proposal_tokens",part(tok,prefix,6)},{"anchor",anchor},{"last_logits",sha(last)},{"target_seq",sha(bs)},{"draft_seq",sha(ds)},{"target_full",sha(bf)},{"draft_full",sha(df)},{"first8K",first}};
            blob(bs);blob(ds);blob(bf);blob(df);float_blob(last);std::ofstream(bundle/"seed.json")<<seed.dump(2)<<'\n';
        } else {
            std::ifstream f(bundle/"seed.json");seed=json::parse(f);need(seed["complete"]&&seed["prefix"]==prefix&&seed["prompt_ids_sha256"]==sha(prompt),"seed inputs");
            bs=read_blob(seed["target_seq"]);ds=read_blob(seed["draft_seq"]);bf=read_blob(seed["target_full"]);df=read_blob(seed["draft_full"]);anchor=seed["anchor"];
        }
        out["seed"]=seed;save();
        auto reset=[&](bool full=false) {if(full){restore_full(t,bf);restore_full(d,df);}else{restore(t,bs);restore(d,ds);}common_speculative_set_state(spec.get(),0,{});common_speculative_begin(spec.get(),0,prompt);need(state(t)==bs&&state(d)==ds,"common prefix state restore");};
        reset(true);out["full_restore_exact"]=true;reset();out["sequence_restore_exact"]=true;save();
        auto target=[&](const Tokens &ids,int pos) {auto b=llama_batch_init(ids.size(),0,1);b.n_tokens=ids.size();for(int i=0;i<b.n_tokens;++i){b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=true;}uint64_t c[counter_count]{};begin();int rc=llama_decode(t,b);llama_synchronize(t);end(c);need(rc==0,"target decode");auto v=logits(t,ids.size());need(common_speculative_process(spec.get(),b),"feature process");llama_synchronize(d);llama_batch_free(b);json counts;for(int i=0;i<counter_count;++i)counts[counter_names[i]]=c[i];return std::make_pair(v,counts);};
        auto record=[&](const std::string &key,json value,const Bytes &final_state,json counts) {blob(final_state);auto description=describe(t,final_state);need(description["metadata_changed_while_describing"].empty(),"state descriptor metadata");out["cases"].push_back({{"key",key},{"value",value},{"state_parts",description},{"counts",counts}});save();std::cerr<<"CASE "<<key<<'\n';};
        for(int wave=0;wave<3;++wave) {
            for(int k:{3,5}){reset();auto row=draft(spec.get(),d,prompt,prefix,anchor,k);blob(state(d));out["draft_checks"].push_back({{"wave",wave},{"k",k},{"result",row}});save();}
            for(int n:{1,3,4,5,6,6,5}) {reset();auto result=target(part(tok,prefix,n),prefix);auto ts=state(t),dd=state(d);float_blob(result.first);blob(dd);record("forward/"+std::to_string(wave)+"/"+std::to_string(out["cases"].size())+"/"+std::to_string(n),{{"n",n},{"ids",part(tok,prefix,n)},{"logits",sha(result.first)},{"target",sha(ts)},{"draft",sha(dd)},{"taps",taps(t)}},ts,result.second);}
            for(auto[n,keep]:std::vector<std::pair<int,int>>{{4,1},{4,2},{4,4},{6,1},{6,3},{6,6}}) {
                reset();auto proposal=draft(spec.get(),d,prompt,prefix,anchor,n-1);need(llama_memory_seq_rm(llama_get_memory(d),0,prefix,-1),"remove noise");Tokens vi{anchor};auto ids=proposal["ids"].get<Tokens>();vi.insert(vi.end(),ids.begin(),ids.end());auto result=target(vi,prefix);auto &v=result.first;auto st=state(t),sd=state(d);int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(t)));int a=std::max_element(v.end()-nv,v.end())-(v.end()-nv);auto next=draft(spec.get(),d,prompt,prefix+n,a,3);
                need(llama_memory_seq_rm(llama_get_memory(t),0,prefix+keep,-1),"target rollback");need(llama_memory_seq_rm(llama_get_memory(d),0,prefix+keep,-1),"draft rollback");common_speculative_accept(spec.get(),0,keep-1);Tokens tail(vi.begin()+keep,vi.end());Floats got;if(!tail.empty())got=apply(t,d,spec.get(),tail,prefix+keep,true);Floats expected(v.begin()+size_t(keep)*nv,v.end());auto next2=draft(spec.get(),d,prompt,prefix+n,a,3);auto replay=state(t),final_state=state(t),dd=state(d);bool ok=got==expected&&next==next2&&replay==st&&final_state==st;out["internal_exact"]=bool(out["internal_exact"])&&ok;
                float_blob(v);float_blob(got);blob(st);blob(sd);blob(replay);blob(dd);
                record("rollback/"+std::to_string(wave)+"/"+std::to_string(n)+"/"+std::to_string(keep),{{"n",n},{"keep",keep},{"ids",vi},{"verify",sha(v)},{"tail",sha(got)},{"target_before",sha(st)},{"target",sha(final_state)},{"draft_before",sha(sd)},{"draft",sha(dd)},{"logits_replay",got==expected},{"next_draft_replay",next==next2},{"target_replay",replay==st&&final_state==st}},final_state,result.second);
            }
        }
        reset();out["base_final_restore_exact"]=true;out["complete"]=true;save();std::cerr<<"SHARED_PREFIX_DONE\n";return 0;
    } catch(const std::exception &e) {out["error"]=e.what();save();std::cerr<<"SHARED_PREFIX_FAIL "<<e.what()<<'\n';return 1;}
}
