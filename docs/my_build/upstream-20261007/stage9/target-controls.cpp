#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <sstream>
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

struct Options {
    std::string mode;
    bool features;
};

static bool precision_control(ggml_tensor * tensor, bool ask, void * data) {
    const auto & options = *static_cast<Options *>(data);
    if (ask && tensor->op == GGML_OP_MUL_MAT &&
            ((options.mode == "float" || options.mode == "strict") || (options.mode == "hc" && std::string(tensor->name).find("hc_mixes-") == 0))) {
        if (tensor->src[0]->type == GGML_TYPE_F32 || tensor->src[0]->type == GGML_TYPE_BF16 || tensor->src[0]->type == GGML_TYPE_F16) {
            ggml_mul_mat_set_prec(tensor, GGML_PREC_F32);
        }
    }
    return false;
}

static void replay(llama_model * model, const std::vector<llama_token> & prompt,
        const std::vector<llama_token> & forced, int width, bool capture, int rollback, const std::string & out, Options options, int switch_at = 0) {
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
    cp.cb_eval = precision_control;
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
    for (int start = 0; start < 64;) {
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
    llama_batch_free(batch);
    llama_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        check(argc == 6, "usage: target-replay MODEL PROMPT FORCED_I32 OUTPUT MODE");
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
        std::istringstream modes(argv[5]);
        for (std::string mode; std::getline(modes, mode, ',');) {
            Options options{mode, false};
            for (int width : {1,2,4}) replay(model,prompt,forced,width,false,0,argv[4],options);
            for (int width : {2,4}) replay(model,prompt,forced,width,false,0,argv[4],options,16);
            if (mode == "default") {
                options.features = true;
                for (int width : {1,2,4}) replay(model,prompt,forced,width,false,0,argv[4],options);
            }
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what());
        return 1;
    }
}
