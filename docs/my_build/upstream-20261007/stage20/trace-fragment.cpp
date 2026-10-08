struct TraceState {
    DownState * down_state;
    bool enabled = true;
    int first_input = 0;
    int prompt_tokens = 0;
    int raw = 0;
    int raw_limit = 0;
    FILE * trace = nullptr;
    FILE * manifest = nullptr;
    std::string directory;
    int event = 0;
};
static void save_tensor(ggml_tensor * tensor, TraceState & state, const std::string & owner, const std::string & role) {
    const std::string file="tensor-"+std::to_string(state.event++)+".bin";
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
    std::ofstream(state.directory+"/"+file,std::ios::binary).write(bytes.data(),bytes.size());
    std::fprintf(state.manifest,
        "{\"file\":\"%s\",\"owner\":\"%s\",\"role\":\"%s\",\"name\":\"%s\",\"op\":\"%s\",\"type\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu],\"bytes\":%zu}\n",
        file.c_str(),owner.c_str(),role.c_str(),tensor->name,ggml_op_name(tensor->op),tensor->type,
        (long long)tensor->ne[0],(long long)tensor->ne[1],(long long)tensor->ne[2],(long long)tensor->ne[3],
        tensor->nb[0],tensor->nb[1],tensor->nb[2],tensor->nb[3],bytes.size());
}
static bool trace_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & state=*static_cast<TraceState *>(data);
    const bool requested=down_precision(tensor,ask,state.down_state);
    if (ask) return requested;
    if (!state.enabled || !state.down_state->precision->decoding || tensor->op!=GGML_OP_FLASH_ATTN_EXT) return true;
    const int layer=std::stoi(std::string(tensor->src[0]->name).substr(2));
    if (layer!=0 && layer!=2 && layer!=20) return true;
    if (layer==0) {
        state.raw=tensor->src[1]->ne[1];
        state.raw_limit=tensor->src[1]->nb[3]/tensor->src[1]->nb[1];
    }
    const auto * mask=tensor->src[3];
    check(mask && mask->type==GGML_TYPE_F16 && mask->ne[2]==1,"unsupported trace mask");
    bool focus=false;
    for (int col=0;col<tensor->src[0]->ne[1];++col) {
        std::vector<ggml_fp16_t> values(mask->ne[0]);
        ggml_backend_tensor_get(mask,values.data(),col*mask->nb[1],values.size()*2);
        int rn=0,cn=0,rmin=-1,rmax=-1,cmin=-1,cmax=-1;
        for (int i=0;i<(int)values.size();++i) if (std::isfinite(ggml_fp16_to_fp32(values[i]))) {
            if (i<state.raw) { ++rn; if (rmin<0) rmin=i; rmax=i; }
            else { ++cn; if (cmin<0) cmin=i-state.raw; cmax=i-state.raw; }
        }
        const int input=state.first_input+col,pos=state.prompt_tokens+input;
        for (int boundary : {3033,3072,3289,3328,6617,6657}) if (pos>=boundary-3 && pos<=boundary+2) focus=true;
        std::fprintf(state.trace,
            "{\"input\":%d,\"output\":%d,\"absolute_position\":%d,\"batch_start\":%d,\"width\":%lld,\"column\":%d,\"layer\":%d,\"raw\":%d,\"raw_limit\":%d,\"compressed\":%lld,\"n_kv_max\":%d,\"raw_visible\":%d,\"raw_first\":%d,\"raw_last\":%d,\"comp_visible\":%d,\"comp_first\":%d,\"comp_last\":%d}\n",
            input,input+1,pos,state.first_input,(long long)tensor->src[0]->ne[1],col,layer,state.raw,state.raw_limit,
            (long long)tensor->src[1]->ne[1]-state.raw,tensor->op_params[4],rn,rmin,rmax,cn,cmin,cmax);
    }
    if (focus) {
        const std::string owner="FA-"+std::to_string(layer);
        save_tensor(tensor,state,owner,"output");
        for (int i=0;i<5;++i) if (tensor->src[i]) save_tensor(tensor->src[i],state,owner,"src"+std::to_string(i));
        std::ofstream params(state.directory+"/"+owner+"-params.json");
        params<<"{\"op_params_u32\":[";
        for (size_t i=0;i<sizeof(tensor->op_params)/4;++i) {
            uint32_t word;std::memcpy(&word,tensor->op_params+i,4);params<<(i?",":"")<<word;
        }
        params<<"]}\n";
    }
    return true;
}
