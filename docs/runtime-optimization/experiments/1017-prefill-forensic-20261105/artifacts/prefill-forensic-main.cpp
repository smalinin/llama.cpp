static std::filesystem::path blob_dir;
static std::string store(const void *ptr,size_t n) {
    auto hash=sha(ptr,n);auto path=blob_dir/(hash+".bin");
    if(!std::filesystem::exists(path)){auto tmp=path.string()+".tmp";{std::ofstream f(tmp,std::ios::binary);f.write(static_cast<const char*>(ptr),n);need(bool(f),"raw blob write");}std::filesystem::rename(tmp,path);}
    return hash;
}
struct TapRead { json hashes;std::vector<std::pair<const float*,size_t>> spans; };
static TapRead read_taps(llama_context*t) {
    uint32_t begin=0,end=0;need(llama_get_embeddings_layer_inp_range(t,&begin,&end)&&end>begin,"tap range");TapRead out;out.hashes={{"begin",begin},{"end",end},{"values",json::array()}};
    for(int lid:{37,38,39}){auto *q=t->get_layer_input_span(lid,begin,end-begin);need(q,"tap data");size_t n=size_t(end-begin)*llama_model_n_embd(llama_get_model(t))*sizeof(float);out.hashes["values"].push_back(sha(q,n));out.spans.emplace_back(q,n);}
    return out;
}
static json describe_state(llama_context*c,const Bytes &data,bool full) {
    size_t start=8;
    if(full){uint32_t length=0;need(data.size()>=4,"global header");std::memcpy(&length,data.data(),4);start=4+length;need(start<=data.size(),"arch length");need(std::string(reinterpret_cast<const char*>(data.data()+4),length)==llm_arch_name(llama_get_model(c)->arch),"arch header");}
    StateDescribe w(data);w.offset=w.meta_begin=start;w.parts.push_back({{"kind","metadata"},{"name","state_header"},{"offset",0},{"bytes",start},{"sha256",sha(data.data(),start)}});llama_get_memory(c)->state_write(w,full?-1:0,0);w.flush();need(w.n_bytes()==data.size(),"descriptor length");return{{"bytes",data.size()},{"sha256",sha(data)},{"parts",w.parts},{"metadata_changed_while_describing",w.bad},{"max_pos",llama_memory_seq_pos_max(llama_get_memory(c),0)}};
}
int main(int argc,char**argv) {
    std::filesystem::path dest=argv[1];blob_dir=std::filesystem::path(argv[3])/"blobs";std::filesystem::create_directories(dest);std::filesystem::create_directories(blob_dir);json out={{"complete",false},{"cases",json::array()}};auto save=[&]{std::ofstream(dest/"results.json")<<out.dump(2)<<'\n';};
    try {
        std::ifstream f(argv[2]);auto input=json::parse(f);auto tok=input["cases"][0]["tokens"].get<Tokens>();common_init();common_params p;std::vector<char*>args{argv[0]};for(int i=4;i<argc;++i)args.push_back(argv[i]);need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"params");p.fit_params=false;p.warmup=false;p.n_outputs_max=p.n_outputs_max_per_seq=6;p.speculative.draft.backend_sampling=false;p.speculative.draft.p_min=0;p.speculative.draft.n_max=5;setenv("LLAMA_DSPARK_CUM_P_MIN","0",1);setenv("LLAMA_DSPARK_EMPTY_COOLDOWN","0",1);unsetenv("LLAMA_DSPARK_GPU_FEATURES");unsetenv("LLAMA_DSPARK_GPU_FEATURES_LOCAL");unsetenv("LLAMA_DSPARK_GPU_FEATURES_ASYNC");unsetenv("LLAMA_DSV41_PREFILL_SKIP");
        llama_backend_init();auto init=common_init_from_params(p);auto*t=init->context();need(t,"target");common_threadpools tp;tp.init(t,p);auto pd=common_base_params_to_speculative(p);auto di=common_speculative_init_from_params(pd,init->model(),t);auto*d=di->context();need(d,"draft");common_threadpools dp;dp.init(d,pd);p.speculative.draft.ctx_tgt=t;p.speculative.draft.ctx_dft=d;common_speculative_ptr spec(common_speculative_init(p.speculative,1));need(bool(spec),"spec");need(llama_set_dflash_prefill_tail(t,d),"tail registration");need(llama_n_rs_seq(t)==5&&t->get_cparams().pipeline_parallel,"fixed context");need(std::string(std::getenv("LLAMA_DSV41_VERIFY_MOE_WIDE"))=="0","wide disabled");std::cerr<<"READY\n";
        for(int repeat=0;repeat<3;++repeat) {
            llama_memory_clear(llama_get_memory(t),true);llama_memory_clear(llama_get_memory(d),true);common_speculative_set_state(spec.get(),0,{});const auto empty_t=sha(state(t)),empty_d=sha(state(d));
            apply(t,d,spec.get(),part(tok,0,8192),0,false,false);
            auto tap=read_taps(t);
            auto ts=state(t);auto tsh=sha(ts);
            auto tf=state(t,true);auto tfh=sha(tf);
            auto ds=state(d);auto dsh=sha(ds);
            auto df=state(d,true);auto dfh=sha(df);
            json value={{"taps",tap.hashes},{"target_seq",tsh},{"target_global",tfh},{"draft_seq",dsh},{"draft_global",dfh}};
            json raw_taps=json::array();for(auto [ptr,n]:tap.spans)raw_taps.push_back(store(ptr,n));
            need(store(ts.data(),ts.size())==tsh&&store(tf.data(),tf.size())==tfh&&store(ds.data(),ds.size())==dsh&&store(df.data(),df.size())==dfh,"owned raw state integrity");
            out["cases"].push_back({{"repeat",repeat},{"value",value},{"raw_tap_hashes",raw_taps},{"tap_raw_matches_original",raw_taps==tap.hashes["values"]},{"empty_target",empty_t},{"empty_draft",empty_d}});save();
            json descriptions={{"target_seq",describe_state(t,ts,false)},{"target_global",describe_state(t,tf,true)},{"draft_seq",describe_state(d,ds,false)},{"draft_global",describe_state(d,df,true)}};
            bool meta=true;for(auto &v:descriptions.items())meta=meta&&v.value()["metadata_changed_while_describing"].empty();out["cases"].back()["descriptors"]=descriptions;out["cases"].back()["metadata_consistent"]=meta;save();std::cerr<<"CAPTURE "<<repeat<<" target="<<tsh<<" metadata="<<meta<<'\n';
        }
        out["complete"]=true;save();std::cerr<<"PREFILL_FORENSIC_DONE\n";return 0;
    }catch(const std::exception&e){out["error"]=e.what();save();std::cerr<<"PREFILL_FORENSIC_FAIL "<<e.what()<<'\n';return 1;}
}
