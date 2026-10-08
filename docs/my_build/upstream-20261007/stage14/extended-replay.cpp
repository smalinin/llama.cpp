#include "diagnostic-callback.h"
using namespace ds14;
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
        for (const std::string mode : {"decode-scalar-fa-upgate-hc-router"}) {
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
