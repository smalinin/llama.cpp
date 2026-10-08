#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void logging(ggml_log_level level, const char * text, void *) {
    if (level <= GGML_LOG_LEVEL_WARN || std::string(text).find("load_tensors:") != std::string::npos) {
        std::fputs(text, stderr);
    }
}

static std::string read_text(const char * path) {
    std::ifstream input(path);
    check(bool(input), "cannot read input");
    return {std::istreambuf_iterator<char>(input), {}};
}


struct PrecisionOptions {
    std::string mode;
    bool features;
    bool decoding = false;
    int replaced = 0;
    int replaced_upgate = 0;
    std::string directory, label;
    std::vector<std::string> captured;
    bool capture_active = false;
    std::string capture_directory;
    FILE * manifest = nullptr;
    int event = 0;
};

static void scalar_attention(ggml_tensor * tensor, PrecisionOptions & options) {
    auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend, "scalar attention backend failed");
    auto * source_q = tensor->src[0];
    auto * source_mask = tensor->src[3];
    check(source_mask && source_mask->ne[2] == 1 && source_q->ne[3] == 1, "unsupported attention shape");
    for (int col = 0; col < source_q->ne[1]; ++col) {
        auto ctx = ggml_init({2*1024*1024, nullptr, true});
        check(ctx, "scalar attention context failed");
        auto borrow = [&](ggml_tensor * src) {
            auto * leaf = ggml_dup_tensor(ctx, src);
            std::memcpy(leaf->nb, src->nb, sizeof(leaf->nb));
            leaf->data = src->data;
            leaf->buffer = src->buffer;
            return leaf;
        };
        auto * q_leaf = borrow(source_q);
        auto * mask_leaf = borrow(source_mask);
        auto * q = ggml_view_4d(ctx, q_leaf, source_q->ne[0], 1, source_q->ne[2], 1,
                source_q->nb[1], source_q->nb[2], source_q->nb[3], col*source_q->nb[1]);
        auto * mask = ggml_view_4d(ctx, mask_leaf, source_mask->ne[0], 1, 1, 1,
                source_mask->nb[1], source_mask->nb[2], source_mask->nb[3], col*source_mask->nb[1]);
        float scale, max_bias, cap;
        std::memcpy(&scale, tensor->op_params, 4);
        std::memcpy(&max_bias, tensor->op_params+1, 4);
        std::memcpy(&cap, tensor->op_params+2, 4);
        auto * y = ggml_flash_attn_ext(ctx, q, borrow(tensor->src[1]), borrow(tensor->src[2]), mask, scale, max_bias, cap);
        if (tensor->src[4]) ggml_flash_attn_ext_add_sinks(y, borrow(tensor->src[4]));
        ggml_flash_attn_ext_set_prec(y, ggml_flash_attn_ext_get_prec(tensor));
        ggml_flash_attn_ext_set_n_kv_max(y, tensor->op_params[4]);
        check(tensor->src[5] == nullptr, "indexed attention not supported by this probe");
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        check(ggml_graph_n_nodes(graph) <= 3, "scalar graph escaped source isolation");
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer, "scalar attention allocation failed");
        check(ggml_backend_graph_compute(backend, graph)==GGML_STATUS_SUCCESS, "scalar attention failed");
        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, out.size()*4);
        ggml_backend_tensor_set(tensor, out.data(), col*out.size()*4, out.size()*4);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ++options.replaced;
    }
    ggml_backend_free(backend);
}

static void scalar_upgate(ggml_tensor * tensor, PrecisionOptions & options) {
    auto * gate_op = tensor->src[0];
    auto * up_op = tensor->src[1];
    check(tensor->op == GGML_OP_GLU && ggml_get_glu_op(tensor) == GGML_GLU_OP_SWIGLU_CLAMP,
          "unsupported routed GLU");
    check(gate_op->op == GGML_OP_MUL_MAT_ID && up_op->op == GGML_OP_MUL_MAT_ID,
          "unsupported routed projections");
    check(gate_op->src[1] == up_op->src[1] && gate_op->src[2] == up_op->src[2],
          "routed projections have different inputs");
    auto * source_x = gate_op->src[1];
    auto * source_ids = gate_op->src[2];
    const int width = source_x->ne[2];
    check(width > 1 && width <= 4 && ggml_is_contiguous(tensor), "unsupported routed batch");
    auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend, "scalar upgate backend failed");
    auto ctx = ggml_init({2*1024*1024, nullptr, true});
    check(ctx, "scalar upgate context failed");
    auto borrow = [&](ggml_tensor * src) {
        auto * leaf = ggml_dup_tensor(ctx, src);
        std::memcpy(leaf->nb, src->nb, sizeof(leaf->nb));
        leaf->data = src->data;
        leaf->buffer = src->buffer;
        return leaf;
    };
    auto * x = borrow(source_x);
    auto * ids = borrow(source_ids);
    auto * gate_weight = borrow(gate_op->src[0]);
    auto * up_weight = borrow(up_op->src[0]);
    float limit;
    std::memcpy(&limit, tensor->op_params+3, 4);
    ggml_tensor * result = nullptr;
    for (int col = 0; col < width; ++col) {
        auto * x_col = ggml_view_3d(ctx, x, x->ne[0], 1, 1, x->nb[1], x->nb[2], col*x->nb[2]);
        auto * ids_col = ggml_view_2d(ctx, ids, ids->ne[0], 1, ids->nb[1], col*ids->nb[1]);
        auto * up = ggml_mul_mat_id(ctx, up_weight, x_col, ids_col);
        auto * gate = ggml_mul_mat_id(ctx, gate_weight, x_col, ids_col);
        auto * y = ggml_swiglu_clamp(ctx, gate, up, limit);
        result = result ? ggml_concat(ctx, result, y, 2) : y;
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    check(ggml_graph_n_nodes(graph) <= 6*width, "scalar upgate escaped source isolation");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer, "scalar upgate allocation failed");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "scalar upgate failed");
    check(ggml_nbytes(result) == ggml_nbytes(tensor), "scalar upgate shape differs");
    std::vector<float> out(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, out.size()*4);
    ggml_backend_tensor_set(tensor, out.data(), 0, out.size()*4);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    options.replaced_upgate += width;
}

static void capture_first(ggml_tensor * tensor, PrecisionOptions & options, const std::string & phase) {
    const std::string name = tensor->name;
    if (name != "ffn_moe_swiglu_limited-2") return;
    const std::string key = name+"-"+phase;
    if (std::find(options.captured.begin(), options.captured.end(), key) != options.captured.end()) return;
    options.captured.push_back(key);
    check(ggml_is_contiguous(tensor) && tensor->type == GGML_TYPE_F32, "unsupported capture");
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*4);
    std::ofstream(options.directory+"/"+options.label+"-"+key+".f32", std::ios::binary)
        .write(reinterpret_cast<const char *>(values.data()), values.size()*4);
}

static bool capture_boundary(const std::string & name) {
    for (int layer = 0; layer < 40; ++layer) {
        for (const std::string prefix : {"hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm", "attn_out", "hc_attn_post", "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights_scaled", "ffn_moe_swiglu_limited", "ffn_moe_out", "ffn_out", "l_last"}) {
            if (name == prefix+"-"+std::to_string(layer)) return true;
        }
    }
    return false;
}

static void save_boundary(ggml_tensor * tensor, PrecisionOptions & options, const std::string & owner, const std::string & role) {
    const std::string file = "tensor-"+std::to_string(options.event++)+".bin";
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::ofstream(options.capture_directory+"/"+file, std::ios::binary).write(bytes.data(), bytes.size());
    std::fprintf(options.manifest,
        "{\"file\":\"%s\",\"owner\":\"%s\",\"role\":\"%s\",\"name\":\"%s\",\"op\":\"%s\",\"type\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu],\"bytes\":%zu}\n",
        file.c_str(), owner.c_str(), role.c_str(), tensor->name, ggml_op_name(tensor->op), tensor->type,
        (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3], bytes.size());
    std::fflush(options.manifest);
}

static bool controlled_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & options = *static_cast<PrecisionOptions *>(data);
    if (!options.decoding) return false;
    if (ask && tensor->op == GGML_OP_MUL_MAT && options.mode.find("decode") == 0) {
        const auto type = tensor->src[0]->type;
        if (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) ggml_mul_mat_set_prec(tensor,GGML_PREC_F32);
    }
    const bool attention = tensor->op == GGML_OP_FLASH_ATTN_EXT;
    const bool upgate = tensor->op == GGML_OP_GLU && std::string(tensor->name).find("ffn_moe_swiglu_limited-") == 0;
    const std::string name = tensor->name;
    const bool boundary = options.capture_active && capture_boundary(name);
    if (ask) return attention || upgate || boundary;
    if (attention && options.mode.find("scalar-fa") != std::string::npos && tensor->src[0]->ne[1] > 1) scalar_attention(tensor, options);
    if (upgate) {
        capture_first(tensor, options, "before");
        if (options.mode.find("upgate") != std::string::npos && tensor->ne[2] > 1) scalar_upgate(tensor, options);
        capture_first(tensor, options, "after");
    }
    if (boundary) {
        save_boundary(tensor, options, name, "output");
        bool early = false;
        for (int layer = 0; layer < 4; ++layer) {
            early |= name == "hc_mixes-"+std::to_string(layer);
            early |= name == "hc_attn_post-"+std::to_string(layer);
            early |= name == "l_last-"+std::to_string(layer);
        }
        if (early && (tensor->op == GGML_OP_DSV4_HC_POST || tensor->op == GGML_OP_MUL_MAT)) {
            const int sources = tensor->op == GGML_OP_DSV4_HC_POST ? 4 : 2;
            for (int i = 0; i < sources; ++i) if (tensor->src[i]) save_boundary(tensor->src[i], options, name, "src"+std::to_string(i));
        }
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
    cp.cb_eval = controlled_precision;
    cp.cb_eval_user_data = &options;
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
    options.capture_directory = out+"/"+label;
    std::filesystem::create_directories(options.capture_directory);
    options.manifest = std::fopen((options.capture_directory+"/tensors.jsonl").c_str(), "w");
    check(options.manifest, "manifest failed");
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
    for (int start = 0; start < 64;) {
        options.capture_active = start == 0;
        batch.n_tokens = std::min(start < switch_at ? 1 : width, 64-start);
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = forced[start+i];
            batch.pos[i] = prompt.size()+start+i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        check(llama_decode(ctx, batch) == 0, "decode failed");
        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);
        start += batch.n_tokens;
    }
    std::printf("DONE %s rows=65 vocab=%d prompt=%zu\n", label.c_str(), vocab, prompt.size());
    std::fflush(stdout);
    std::printf("REPLACED %s %d queries\n", label.c_str(), options.replaced);
    std::printf("REPLACED_UPGATE %s %d token-layers\n", label.c_str(), options.replaced_upgate);
    std::fclose(options.manifest);
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
        check(input.gcount()>=65*4, "forced tokens incomplete");
        for (const std::string mode : {"decode-scalar-fa-upgate-boundaries"}) {
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
