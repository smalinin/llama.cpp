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
