#include "diagnostic-callback.h"
using namespace ds14;

struct DownInput {
    std::vector<float> hidden,weights;
    std::vector<int32_t> ids;
};
struct DownState {
    PrecisionOptions * precision;
    std::map<int,DownInput> inputs;
    std::string capture_prefix;
};
static void scalar_routed_down(ggml_tensor * tensor, PrecisionOptions & options, const DownInput & input) {
    auto * view=tensor->src[1];
    check(view && view->op==GGML_OP_VIEW, "missing last expert view");
    auto * weighted=view->src[0];
    check(weighted && weighted->op==GGML_OP_MUL, "missing expert weights");
    auto * down=weighted->src[0];
    check(down && down->op==GGML_OP_MUL_MAT_ID, "missing routed down");
    auto * hidden=down->src[1];
    auto * ids=down->src[2];
    auto * weights=weighted->src[1];
    const int width=tensor->ne[1];
    check(width>1 && width<=4 && hidden->ne[2]==width && hidden->ne[1]==6 &&
            weights->ne[0]==1 && weights->ne[1]==6 && weights->ne[2]==width && ggml_is_contiguous(tensor), "unsupported down shape");
    auto device=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend=ggml_backend_dev_init(device,nullptr);
    check(backend,"scalar down backend failed");
    auto ctx=ggml_init({2*1024*1024,nullptr,true});
    check(ctx,"scalar down context failed");
    auto borrow=[&](ggml_tensor * src) {
        auto * leaf=ggml_dup_tensor(ctx,src);
        std::memcpy(leaf->nb,src->nb,sizeof(leaf->nb));leaf->data=src->data;leaf->buffer=src->buffer;
        return leaf;
    };
    auto * h=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hidden->ne[0],6,width);
    auto * all_ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,width);
    auto * all_weights=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,6,width);
    auto * w=borrow(down->src[0]);
    auto * graph=ggml_new_graph(ctx);
    ggml_tensor * result=nullptr;
    for (int col=0;col<width;++col) {
        auto * hc=ggml_view_3d(ctx,h,h->ne[0],h->ne[1],1,h->nb[1],h->nb[2],col*h->nb[2]);
        auto * ic=ggml_view_2d(ctx,all_ids,6,1,all_ids->nb[1],col*all_ids->nb[1]);
        auto * wc=ggml_view_3d(ctx,all_weights,1,6,1,all_weights->nb[1],all_weights->nb[2],col*all_weights->nb[2]);
        auto * dc=ggml_mul_mat_id(ctx,w,hc,ic);
        auto * ew=ggml_mul(ctx,dc,wc);
        ggml_build_forward_expand(graph,ew);
        std::vector<ggml_tensor *> views;
        for (int i=0;i<6;++i) {
            auto * v=ggml_view_2d(ctx,ew,dc->ne[0],1,ew->nb[2],i*ew->nb[1]);
            ggml_build_forward_expand(graph,v);views.push_back(v);
        }
        auto * y=views[0];
        for (int i=1;i<6;++i) {y=ggml_add(ctx,y,views[i]);ggml_build_forward_expand(graph,y);}
        result=result?ggml_concat(ctx,result,y,1):y;
    }
    ggml_build_forward_expand(graph,result);
    check(ggml_graph_n_nodes(graph)<=20*width,"scalar down escaped source isolation");
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    check(buffer,"scalar down allocation failed");
    check(input.hidden.size()==size_t(hidden->ne[0]*6*width) && input.ids.size()==size_t(6*width) && input.weights.size()==size_t(6*width),"cached down inputs incomplete");
    ggml_backend_tensor_set(h,input.hidden.data(),0,input.hidden.size()*4);
    ggml_backend_tensor_set(all_ids,input.ids.data(),0,input.ids.size()*4);
    ggml_backend_tensor_set(all_weights,input.weights.data(),0,input.weights.size()*4);
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"scalar down compute failed");
    check(ggml_nbytes(result)==ggml_nbytes(tensor),"scalar down result shape differs");
    std::vector<float> values(ggml_nelements(result));
    ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
    ggml_backend_tensor_set(tensor,values.data(),0,values.size()*4);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    options.matmul_counts["scalar-down-"+std::string(tensor->name)]+=width;
}

static bool down_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & state=*static_cast<DownState *>(data);
    auto & options=*state.precision;
    const bool requested=controlled_precision(tensor,ask,&options);
    if (!options.decoding) return requested;
    const std::string name=tensor->name;
    const bool weight=name.find("ffn_moe_weights_scaled-")==0;
    const bool hidden=name.find("ffn_moe_swiglu_limited-")==0;
    const bool output=name.find("ffn_moe_out-")==0;
    const bool capture=!state.capture_prefix.empty() && (name=="ffn_moe_out-0" || name=="ffn_out-0");
    if (ask) return requested || weight || output || capture;
    if (weight || hidden || output) {
        const int layer=std::stoi(name.substr(name.rfind('-')+1));
        auto & input=state.inputs[layer];
        if (weight && tensor->ne[2]>1) {
            check(ggml_is_contiguous(tensor),"non-contiguous expert weights");
            input.weights.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor,input.weights.data(),0,input.weights.size()*4);
        }
        if (hidden && tensor->ne[2]>1) {
            check(ggml_is_contiguous(tensor),"non-contiguous hidden");
            input.hidden.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor,input.hidden.data(),0,input.hidden.size()*4);
            auto * ids=tensor->src[0]->src[2];
            check(ids && ids->type==GGML_TYPE_I32 && ids->ne[0]==6,"unsupported cached IDs");
            input.ids.resize(6*tensor->ne[2]);
            for (int col=0;col<tensor->ne[2];++col) ggml_backend_tensor_get(ids,input.ids.data()+6*col,col*ids->nb[1],6*4);
        }
        if (output && tensor->ne[1]>1) scalar_routed_down(tensor,options,input);
    }
    if (capture) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);
        std::ofstream(state.capture_prefix+"-"+name+".f32",std::ios::binary).write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    }
    return true;
}


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

static void precision_replay(llama_model * model, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & forced, int width, bool capture, int rollback, const std::string & out, PrecisionOptions options, int switch_at = 0) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 12;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 4;
    if (options.mode == "strict") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (options.mode == "f32kv") cp.type_k = cp.type_v = GGML_TYPE_F32;
    DownState state{&options};
    CaptureState capture_state{&state};
    cp.cb_eval = capture_precision;
    cp.cb_eval_user_data = &capture_state;
    cp.kv_unified = false;
    cp.n_rs_seq = rollback;
    cp.swa_full = false;
    cp.no_perf = false;
    auto * ctx = llama_init_from_model(model, cp);
    check(ctx, "context failed");
    auto * memory = llama_get_memory(ctx);
    llama_memory_clear(memory, true);
    std::vector<llama_token> probe(2, 0);
    check(llama_decode(ctx, llama_batch_get_one(probe.data(), probe.size())) == 0, "probe failed");
    if (!rollback) llama_memory_seq_rm(memory, 0, 1, -1);
    llama_memory_clear(memory, true);
    llama_synchronize(ctx);
    if (options.features) {
        for (uint32_t layer = 0; layer <= 40; ++layer) llama_set_embeddings_layer_inp(ctx, layer, true);
    }
    auto batch = llama_batch_init(2048, 0, 1);
    int prompt_start = 0;
    for (int prompt_end : {(int) prompt.size()-516, (int) prompt.size()-4, (int) prompt.size()}) {
        batch.n_tokens = prompt_end-prompt_start;
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = prompt[prompt_start+i];
            batch.pos[i] = prompt_start+i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = prompt_start+i+1 == (int) prompt.size();
        }
        check(llama_decode(ctx, batch) == 0, "prefill failed");
        prompt_start = prompt_end;
    }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::string label = options.mode + (options.features ? "-features" : "") + "-w" + std::to_string(width) +
        (switch_at ? "-switch" + std::to_string(switch_at) : "");
    options.directory = out;
    options.label = label;
    std::ofstream features_file;
    if (options.features) features_file.open(out + "/" + label + "-features.f32", std::ios::binary);
    std::ofstream logits_file(out + "/" + label + "-logits.f32", std::ios::binary);
    std::ofstream rows(out + "/" + label + "-rows.jsonl");
    auto record = [&](int index, int row) {
        const float * logits = llama_get_logits_ith(ctx, row);
        check(logits, "missing logits");
        if (options.features) {
            for (uint32_t layer = 0; layer <= 40; ++layer) {
                const float * features = llama_get_embeddings_layer_inp(ctx, layer) + (row < 0 ? batch.n_tokens - 1 : row)*5120;
                check(features, "missing features");
                features_file.write(reinterpret_cast<const char *>(features), 5120*sizeof(float));
            }
        }
        logits_file.write(reinterpret_cast<const char *>(logits), vocab*sizeof(float));
        int best = 0, second = 1;
        if (logits[second] > logits[best]) std::swap(best, second);
        for (int i = 2; i < vocab; ++i) {
            check(std::isfinite(logits[i]), "nonfinite logit");
            if (logits[i] > logits[best]) { second = best; best = i; }
            else if (logits[i] > logits[second]) second = i;
        }
        rows << "{\"index\":" << index << ",\"argmax\":" << best << ",\"second\":" << second
             << ",\"gap\":" << logits[best]-logits[second] << ",\"reference\":" << forced[index] << "}\n";
    };
    record(0, -1);
    options.decoding = true;
    for (int start = 0; start < 94;) {
        batch.n_tokens = std::min(start < switch_at ? 1 : width, 94-start);
        capture_state.active=start<20;
        if (capture_state.active) {
            const int target=start;
            capture_state.directory=out+"/"+label+"-input"+std::to_string(target);
            std::filesystem::create_directories(capture_state.directory);
            capture_state.manifest=std::fopen((capture_state.directory+"/tensors.jsonl").c_str(),"w");
            check(capture_state.manifest,"capture manifest failed");
            std::ofstream meta(capture_state.directory+"/alignment.json");
            meta<<"{\"input_index\":"<<target<<",\"output_index\":"<<target+1
                <<",\"batch_start\":"<<start<<",\"width\":"<<batch.n_tokens
                <<",\"column\":"<<target-start<<"}\n";
            capture_state.event=0;
        }
        const int target=start==0?0:86;
        state.capture_prefix=(start==0 || (start<=86 && start+batch.n_tokens>86)) ? out+"/"+label+"-input"+std::to_string(target):"";
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = forced[start+i];
            batch.pos[i] = prompt.size()+start+i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        check(llama_decode(ctx, batch) == 0, "decode failed");
        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);
        if (capture_state.active) {std::fclose(capture_state.manifest);capture_state.manifest=nullptr;}
        start += batch.n_tokens;
    }
    std::printf("DONE %s rows=95 vocab=%d prompt=%zu\n", label.c_str(), vocab, prompt.size());
    std::fflush(stdout);
    std::printf("REPLACED %s %d queries\n", label.c_str(), options.replaced);
    std::printf("REPLACED_UPGATE %s %d token-layers\n", label.c_str(), options.replaced_upgate);
    std::ofstream counts(out+"/"+label+"-matmul-counts.json");
    counts << "{";
    bool first = true;
    for (const auto & item : options.matmul_counts) {
        counts << (first ? "" : ",") << "\"" << item.first << "\":" << item.second;
        first = false;
    }
    counts << "}\n";
    llama_batch_free(batch);
    llama_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        check(argc == 5, "usage: target-replay MODEL PROMPT FORCED_I32 OUTPUT");
        check(!std::filesystem::exists(argv[4]), "output exists");
        std::filesystem::create_directories(argv[4]);
        llama_backend_init();
        ggml_backend_load_all();
        llama_log_set(logging, nullptr);
        auto mp = llama_model_default_params();
        std::vector<float> split(llama_max_devices(), 0.0f);
        for (int i = 0; i < 6; ++i) split[i] = i == 5 ? 0.4f : 1.0f;
        mp.n_gpu_layers = 99;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        mp.tensor_split = split.data();
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode = LLAMA_LAZY_MODE_AUTO;
        auto * model = llama_model_load_from_file(argv[1], mp);
        check(model, "model load failed");
        std::ifstream maps("/proc/self/maps");
        std::ofstream loaded(std::string(argv[4])+"/loaded-libraries.txt");
        for (std::string line; std::getline(maps,line);) {
            if (line.find("/libllama")!=std::string::npos || line.find("/libggml")!=std::string::npos) loaded<<line<<'\n';
        }
        const std::string text = read_text(argv[2]);
        auto * vocab = llama_model_get_vocab(model);
        const int n = -llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
        std::vector<llama_token> prompt(n);
        check(llama_tokenize(vocab,text.data(),text.size(),prompt.data(),n,true,true)==n, "tokenize failed");
        std::ifstream input(argv[3],std::ios::binary);
        std::vector<llama_token> forced(128);
        input.read(reinterpret_cast<char *>(forced.data()),forced.size()*sizeof(llama_token));
        check(input.gcount()>=95*4, "forced tokens incomplete");
        for (const std::string mode : {"decode-scalar-fa-upgate-hc-router-down"}) {
            PrecisionOptions options{mode, false};
            for (int width : {1,2,4}) precision_replay(model,prompt,forced,width,false,0,argv[4],options);
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what());
        return 1;
    }
}
