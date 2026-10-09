#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
int main(int argc, char ** argv) {
    try {
        check(argc == 4, "usage: restore-replay model unified output-directory");
        ggml_backend_load_all();
        auto mp = llama_model_default_params();mp.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], mp);check(model, "model");
        auto cp = llama_context_default_params();
        cp.n_ctx = 1536;cp.n_batch = 192;cp.n_ubatch = 64;cp.n_seq_max = 3;
        cp.kv_unified = std::atoi(argv[2]);cp.n_threads = 2;cp.n_threads_batch = 2;cp.n_rs_seq = 8;
        auto * ctx = llama_init_from_model(model, cp);check(ctx, "context");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<int> pos(3, 0);int total_calls = 0;
        auto decode = [&](const std::vector<int> & seqs, int n, FILE * out) {
            auto batch = llama_batch_init(n*seqs.size(), 0, 1);int i = 0;
            for (int seq : seqs) for (int j = 0; j < n; ++j, ++i) {
                const int p = pos[seq]++;
                batch.token[i] = (seq*17 + p*3 + 1)%vocab;batch.pos[i] = p;
                batch.n_seq_id[i] = 1;batch.seq_id[i][0] = seq;batch.logits[i] = out != nullptr || j + 1 == n;
            }
            batch.n_tokens = i;check(llama_decode(ctx, batch) == 0, "decode");
            for (int row = 0; out && row < i; ++row) {
                const float * logits = llama_get_logits_ith(ctx, row);check(logits, "logits");
                check(std::fwrite(logits, sizeof(float), vocab, out) == size_t(vocab), "write");
            }
            llama_batch_free(batch);++total_calls;return i;
        };
        auto snapshot = [&]() {
            std::vector<std::vector<uint8_t>> states(3);
            for (int seq = 0; seq < 3; ++seq) {
                const size_t n = llama_state_seq_get_size(ctx, seq);check(n > 0, "state size");states[seq].resize(n);
                check(llama_state_seq_get_data(ctx, states[seq].data(), n, seq) == n, "state get");
            }
            return states;
        };
        auto restore = [&](const std::vector<std::vector<uint8_t>> & states, const std::vector<int> & positions) {
            for (int seq = 0; seq < 3; ++seq) {
                check(llama_state_seq_set_data(ctx, states[seq].data(), states[seq].size(), seq) == states[seq].size(), "state set");
            }
            pos = positions;
        };
        auto record = [&](const char * name, const std::vector<std::vector<int>> & schedule) {
            const std::string path = std::string(argv[3]) + "/" + name + ".f32";
            FILE * out = std::fopen(path.c_str(), "wb");check(out, "output");int rows = 0;
            for (const auto & seqs : schedule) rows += decode(seqs, 1, out);
            std::fclose(out);std::printf("{\"phase\":\"%s\",\"rows\":%d,\"vocab\":%d,\"calls_total\":%d}\n",name,rows,vocab,total_calls);std::fflush(stdout);
        };
        decode({0},96,nullptr);decode({1},128,nullptr);decode({2},160,nullptr);
        const auto states = snapshot();const auto prefix = pos;
        const std::vector<std::vector<int>> schedule = {{0,1,2},{0},{2},{0,2},{1},{0,1,2},{0,1,2},{1},{0,2},{0,1,2}};
        record("resident",schedule);
        restore(states,prefix);record("restore1",schedule);
        restore(states,prefix);record("restore2",schedule);
        llama_free(ctx);ctx = llama_init_from_model(model, cp);check(ctx, "new context");
        restore(states,prefix);record("new-context",schedule);
        restore(states,prefix);decode({0,1,2},1,nullptr);
        const auto checkpoint = snapshot();const auto tail_start = pos;
        const std::vector<std::vector<int>> tail = {{0,1,2},{0,1,2},{0,1,2}};
        record("tail-resident",tail);
        for (int seq = 0; seq < 3; ++seq) check(llama_memory_seq_rm(llama_get_memory(ctx),seq,tail_start[seq],-1),"tail rollback");
        pos = tail_start;record("tail-rollback",tail);
        restore(checkpoint,tail_start);record("tail-restore",tail);
        llama_free(ctx);llama_model_free(model);llama_backend_free();
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what());return 2; }
}
