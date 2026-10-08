struct CaptureState {
    DownState * down_state;
    bool active = false;
    std::string directory;
    FILE * manifest = nullptr;
    int event = 0;
};
static bool capture_boundary(const std::string & name) {
    for (int layer : {2}) {
        for (const std::string prefix : {"attn_norm", "qr", "q", "kv", "comp_state_kv", "comp_state_score", "comp_kv", "comp_kv_rot", "idx_top_k", "attn_out_raw", "attn_derope", "attn_wo_a", "attn_out"}) {
            if (name == prefix+"-"+std::to_string(layer)) return true;
        }
    }
    return false;
}
static void save_boundary(ggml_tensor * tensor, CaptureState & state, const std::string & owner, const std::string & role) {
    const std::string file = "tensor-"+std::to_string(state.event++)+".bin";
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::ofstream(state.directory+"/"+file, std::ios::binary).write(bytes.data(), bytes.size());
    std::fprintf(state.manifest,
        "{\"file\":\"%s\",\"owner\":\"%s\",\"role\":\"%s\",\"name\":\"%s\",\"op\":\"%s\",\"type\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu],\"bytes\":%zu}\n",
        file.c_str(), owner.c_str(), role.c_str(), tensor->name, ggml_op_name(tensor->op), tensor->type,
        (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3], bytes.size());
    std::fflush(state.manifest);
}
static bool capture_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & state = *static_cast<CaptureState *>(data);
    const bool requested = down_precision(tensor, ask, state.down_state);
    const std::string name = tensor->name;
    const bool fa = state.active && tensor->op==GGML_OP_FLASH_ATTN_EXT && std::string(tensor->src[0]->name).find("q-2 ")==0;
    const bool boundary = state.active && capture_boundary(name);
    if (ask) return requested || boundary || fa;
    if (boundary) {
        save_boundary(tensor, state, name, "output");
        if ((name=="comp_state_kv-2" || name=="comp_state_score-2" || name=="attn_wo_a-2") && tensor->src[1]) {
            save_boundary(tensor->src[1],state,name,"src1");
        }
    }
    if (fa) {
        save_boundary(tensor,state,"FA-2","output");
        for (int i=0;i<5;++i) if (tensor->src[i]) save_boundary(tensor->src[i],state,"FA-2","src"+std::to_string(i));
        std::ofstream params(state.directory+"/fa-params.json");
        float scale,bias,cap;
        std::memcpy(&scale,tensor->op_params,4);std::memcpy(&bias,tensor->op_params+1,4);std::memcpy(&cap,tensor->op_params+2,4);
        params<<"{\"scale\":"<<std::to_string(scale)<<",\"max_bias\":"<<std::to_string(bias)<<",\"cap\":"<<std::to_string(cap)
              <<",\"n_kv_max\":"<<tensor->op_params[4]<<",\"precision\":"<<ggml_flash_attn_ext_get_prec(tensor)<<"}\n";
    }
    return true;
}
