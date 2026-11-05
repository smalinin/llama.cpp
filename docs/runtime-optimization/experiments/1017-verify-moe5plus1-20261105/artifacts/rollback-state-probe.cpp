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
int main(int argc,char**argv){json out={{"complete",false},{"cases",json::array()},{"faults",json::array()}};std::filesystem::path dest=argv[1];std::filesystem::create_directories(dest);auto save=[&]{std::ofstream(dest/"results.json")<<out.dump(2)<<'\n';};try{std::ifstream f(argv[2]);auto input=json::parse(f);input["cases"].erase(input["cases"].begin()+1,input["cases"].end());out["selector"]=std::getenv("LLAMA_DSV41_VERIFY_MOE_WIDE")?std::getenv("LLAMA_DSV41_VERIFY_MOE_WIDE"):"unset";common_init();common_params p;std::vector<char*>args{argv[0]};for(int i=3;i<argc;++i)args.push_back(argv[i]);need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"params");p.fit_params=false;p.warmup=false;p.n_outputs_max=p.n_outputs_max_per_seq=6;p.speculative.draft.backend_sampling=false;p.speculative.draft.p_min=0;p.speculative.draft.n_max=5;setenv("LLAMA_DSPARK_CUM_P_MIN","0",1);setenv("LLAMA_DSPARK_EMPTY_COOLDOWN","0",1);unsetenv("LLAMA_DSPARK_GPU_FEATURES");unsetenv("LLAMA_DSPARK_GPU_FEATURES_LOCAL");unsetenv("LLAMA_DSPARK_GPU_FEATURES_ASYNC");unsetenv("LLAMA_DSV41_PREFILL_SKIP");llama_backend_init();auto init=common_init_from_params(p);auto*t=init->context();need(t,"target");common_threadpools tp;tp.init(t,p);auto pd=common_base_params_to_speculative(p);auto di=common_speculative_init_from_params(pd,init->model(),t);auto*d=di->context();need(d,"draft");common_threadpools dp;dp.init(d,pd);p.speculative.draft.ctx_tgt=t;p.speculative.draft.ctx_dft=d;common_speculative_ptr spec(common_speculative_init(p.speculative,1));need(bool(spec),"spec");need(llama_set_dflash_prefill_tail(t,d),"tail registration");std::cerr<<"READY\n";int nv=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(t)));

for(auto&corpus:input["cases"]){std::string id=corpus["id"];auto tok=corpus["tokens"].get<Tokens>();std::map<std::string,json>refs;
for(int mode:{0}){llama_memory_clear(llama_get_memory(t),true);llama_memory_clear(llama_get_memory(d),true);common_speculative_set_state(spec.get(),0,{});const auto empty_t=sha(state(t)),empty_d=sha(state(d));
auto record=[&](std::string key,json row){if(mode==0)refs[key]=row;else if(refs.at(key)!=row){out["mismatch"]={{"corpus",id},{"mode",mode},{"key",key},{"reference",refs.at(key)},{"actual",row}};save();throw std::runtime_error("long mode mismatch");}out["cases"].push_back({{"corpus",id},{"mode",mode},{"key",key},{"result",row}});save();std::cerr<<"EXACT "<<id<<' '<<mode<<' '<<key<<'\n';};
json first;Bytes at90,ad90;for(int pos=0;pos<81920;pos+=8192){apply(t,d,spec.get(),part(tok,pos,8192),pos,false,false);if(pos==0)first={{"taps",taps(t)},{"target_seq",sha(state(t))},{"target_global",sha(state(t,true))},{"draft_seq",sha(state(d))},{"draft_global",sha(state(d,true))}};out["progress"]={{"corpus",id},{"mode",mode},{"prefix",pos+8192}};save();std::cerr<<"PREFILL "<<id<<' '<<mode<<' '<<pos+8192<<'\n';}at90=state(t);ad90=state(d);
apply(t,d,spec.get(),part(tok,81920,7676),81920,false,false);auto ca=partial(t),cda=state(d);json ck={{"A_target",sha(state(t))},{"A_draft",sha(state(d))},{"A_partial",sha(ca)}};apply(t,d,spec.get(),part(tok,89596,512),89596,false,false);auto cb=partial(t),cdb=state(d);ck["B_target"]=sha(state(t));ck["B_draft"]=sha(state(d));ck["B_partial"]=sha(cb);auto last=apply(t,d,spec.get(),part(tok,90108,4),90108);auto bs=state(t),ds=state(d),bf=state(t,true),df=state(d,true),bp=partial(t),dpart=partial(d);record("prefix90112",{{"taps",taps(t)},{"first8K",first},{"checkpoints",ck},{"last_logits",sha(last)},{"target_seq",sha(bs)},{"target_global",sha(bf)},{"draft_seq",sha(ds)},{"draft_global",sha(df)}});
Tokens prompt=part(tok,0,90112);int anchor=std::max_element(last.end()-nv,last.end())-(last.end()-nv);common_speculative_begin(spec.get(),0,prompt);json dr=json::array();for(int k:{3,5}){restore(d,ds);dr.push_back(draft(spec.get(),d,prompt,90112,anchor,k));}record("draftK3K5",dr);
restore(t,bs);restore(d,ds);auto verify=apply(t,d,spec.get(),part(tok,90112,6),90112,true);json rr={{"logits",sha(verify)}};
restore_full(t,bf);restore_full(d,df);need(apply(t,d,spec.get(),part(tok,90112,6),90112,true)==verify,"full replay");rr["full_t"]=sha(state(t));rr["full_d"]=sha(state(d));
restore(t,bs);restore(d,ds);need(apply(t,d,spec.get(),part(tok,90112,6),90112,true)==verify,"seq replay");rr["seq_t"]=sha(state(t));rr["seq_d"]=sha(state(d));
restore_partial(t,bp);restore_partial(d,dpart);need(apply(t,d,spec.get(),part(tok,90112,6),90112,true)==verify,"partial replay");rr["partial_t"]=sha(state(t));rr["partial_d"]=sha(state(d));
restore_partial(t,ca);restore(d,cda);apply(t,d,spec.get(),part(tok,89596,512),89596,false,false);need(apply(t,d,spec.get(),part(tok,90108,4),90108)==last,"checkpoint A");rr["A_t"]=sha(state(t));rr["A_d"]=sha(state(d));
restore_partial(t,cb);restore(d,cdb);need(apply(t,d,spec.get(),part(tok,90108,4),90108)==last,"checkpoint B");rr["B_t"]=sha(state(t));rr["B_d"]=sha(state(d));
restore(t,at90);restore(d,ad90);apply(t,d,spec.get(),part(tok,81920,7676),81920,false,false);apply(t,d,spec.get(),part(tok,89596,512),89596,false,false);need(apply(t,d,spec.get(),part(tok,90108,4),90108)==last,"80K resume");rr["resume_t"]=sha(state(t));rr["resume_d"]=sha(state(d));record("restores",rr);
json rolls=json::array();for(auto[n,keep]:std::vector<std::pair<int,int>>{{4,1},{4,2},{4,4},{6,1},{6,3},{6,6}}){restore(t,bs);restore(d,ds);auto proposal=draft(spec.get(),d,prompt,90112,anchor,n-1);need(llama_memory_seq_rm(llama_get_memory(d),0,90112,-1),"remove noise");Tokens vi{anchor};auto ids=proposal["ids"].get<Tokens>();vi.insert(vi.end(),ids.begin(),ids.end());auto v=apply(t,d,spec.get(),vi,90112,true);auto st=state(t),sd=state(d);auto before_desc=describe(t,st);json phases={{"before_next",position(t)}};int a=std::max_element(v.end()-nv,v.end())-(v.end()-nv);auto next=draft(spec.get(),d,prompt,90112+n,a,3);phases["after_next"]=position(t);need(llama_memory_seq_rm(llama_get_memory(t),0,90112+keep,-1),"target rollback");need(llama_memory_seq_rm(llama_get_memory(d),0,90112+keep,-1),"draft rollback");phases["after_rm"]=position(t);common_speculative_accept(spec.get(),0,keep-1);Tokens tail(vi.begin()+keep,vi.end());Floats got;if(!tail.empty())got=apply(t,d,spec.get(),tail,90112+keep,true);Floats expected(v.begin()+size_t(keep)*nv,v.end());need(got==expected,"rollback logits");phases["after_tail"]=position(t);auto next2=draft(spec.get(),d,prompt,90112+n,a,3);need(next==next2,"next draft rollback");auto replay_state=state(t),final_state=state(t);auto after_desc=describe(t,final_state);phases["after_next2"]=position(t);out["state_debug"].push_back({{"n",n},{"keep",keep},{"ids",vi},{"phases",phases},{"before",before_desc},{"after",after_desc},{"difference",byte_diff(st,final_state)},{"first_vs_second_state_read_equal",replay_state==final_state}});rolls.push_back({{"n",n},{"keep",keep},{"verify",sha(v)},{"tail",sha(got)},{"target_replay",replay_state==st},{"draft_replay",next==next2},{"target",sha(final_state)},{"draft",sha(state(d))}});}record("rollback",rolls);}}out["complete"]=true;out.erase("progress");save();std::cerr<<"ROLLBACK_PROBE_DONE\n";return 0;}catch(const std::exception&e){out["error"]=e.what();save();std::cerr<<"PROBE_FAIL "<<e.what()<<'\n';return 1;}}
