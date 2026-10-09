
struct CaptureState {
    bool active = false;
    std::string directory;
    FILE * manifest = nullptr;
    int event = 0;
};
static bool capture_boundary(const std::string & name) {
    for (int layer = 0; layer < 40; ++layer) {
        for (const std::string prefix : {"hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm", "attn_out", "hc_attn_post", "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights_scaled", "ffn_moe_swiglu_limited", "ffn_moe_out", "ffn_out", "l_last"}) {
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
