#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
struct counters { int full = 0; int incremental = 0; };
static bool callback(ggml_tensor * t, bool ask, void * data) {
    if (ask) {
        auto & c = *(counters *) data;
        if (std::strncmp(t->name, "indexer_pool_members-", 21) == 0) ++c.full;
        if (std::strncmp(t->name, "indexer_pool_update_members-", 28) == 0) ++c.incremental;
    }
    return false;
}
int main(int argc, char ** argv) {
    try {
        check(argc == 4, "usage: pool-replay model unified output");
        ggml_backend_load_all();
        auto mp = llama_model_default_params();mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], mp);check(model, "load model");
        auto cp = llama_context_default_params();
        cp.n_ctx = 1536;cp.n_batch = 192;cp.n_ubatch = 64;cp.n_seq_max = 3;
        cp.kv_unified = std::atoi(argv[2]);cp.n_threads = 2;cp.n_threads_batch = 2;cp.n_rs_seq = 8;
        counters count;cp.cb_eval = callback;cp.cb_eval_user_data = &count;
        llama_context * ctx = llama_init_from_model(model, cp);check(ctx, "create context");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        FILE * out = std::fopen(argv[3], "wb");check(out, "open output");
        std::vector<int> positions(3, 0);int rows = 0, calls = 0;
        auto decode = [&](const std::vector<int> & seqs, int n, bool prefill) {
            auto batch = llama_batch_init(n*seqs.size(), 0, 1);int i = 0;
            for (int seq : seqs) {
                for (int j = 0; j < n; ++j, ++i) {
                    const int pos = positions[seq]++;
                    batch.token[i] = (seq*17 + pos*3 + 1)%vocab;batch.pos[i] = pos;
                    batch.n_seq_id[i] = 1;batch.seq_id[i][0] = seq;batch.logits[i] = !prefill || j + 1 == n;
                }
            }
            batch.n_tokens = i;check(llama_decode(ctx, batch) == 0, "decode");
            for (int j = 0; j < i; ++j) if (batch.logits[j]) {
                const float * logits = llama_get_logits_ith(ctx, j);check(logits, "logits");
                check(std::fwrite(logits, sizeof(float), vocab, out) == size_t(vocab), "write logits");++rows;
            }
            llama_batch_free(batch);++calls;
            std::printf("{\"call\":%d,\"rows\":%d,\"full\":%d,\"incremental\":%d}\n", calls, rows, count.full, count.incremental);std::fflush(stdout);
        };
        decode({0}, 96, true);decode({1}, 128, true);decode({2}, 160, true);
        decode({0,1,2}, 1, false);
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (const auto & seqs : std::vector<std::vector<int>>{{0},{2},{0,2},{1},{0,1,2}}) decode(seqs, 1, false);
        }
        decode({0,1,2}, 4, false);
        for (int seq = 0; seq < 3; ++seq) {
            positions[seq] -= 3;
            check(llama_memory_seq_rm(llama_get_memory(ctx), seq, positions[seq], -1), "rollback");
        }
        decode({0,1,2}, 1, false);decode({0,1,2}, 1, false);
        for (int seq = 0; seq < 3; ++seq) check(llama_memory_seq_rm(llama_get_memory(ctx), seq, 0, -1), "clear sequence");
        positions.assign(3, 0);decode({0}, 32, true);decode({1}, 48, true);decode({2}, 64, true);
        decode({0,1,2}, 1, false);
        std::fclose(out);llama_free(ctx);llama_model_free(model);llama_backend_free();
        std::printf("{\"done\":true,\"calls\":%d,\"rows\":%d,\"vocab\":%d,\"full\":%d,\"incremental\":%d}\n",calls,rows,vocab,count.full,count.incremental);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what());return 2; }
}
