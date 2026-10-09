#define main test_archs_main
#include "/home/sergei/Github/llama.cpp/tests/test-llama-archs.cpp"
#undef main

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    try {
        llama_backend_init();
        auto metadata = get_gguf_ctx(LLM_ARCH_GLM5NEXT, true, true);
        auto initial = get_model_and_ctx(metadata.get(), nullptr, 1234, {}, LLAMA_SPLIT_MODE_LAYER,
                false, LLAMA_CONTEXT_TYPE_MTP);
        initial.second.reset();
        auto cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;cp.n_ctx = 768;cp.n_batch = 64;cp.n_ubatch = 64;
        cp.n_seq_max = std::atoi(argv[2]);cp.kv_unified = std::atoi(argv[1]);cp.n_rs_seq = 8;
        cp.n_threads = 2;cp.n_threads_batch = 2;
        mtp_indexer_eval_count counts;cp.cb_eval = count_mtp_indexer_score;cp.cb_eval_user_data = &counts;
        llama_context_ptr ctx(llama_init_from_model(initial.first.get(), cp));
        if (!ctx) throw std::runtime_error("context");
        const int embd = llama_model_n_embd(initial.first.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(initial.first.get()));
        auto batch = llama_batch_init(64, 0, 1);
        std::vector<float> hidden(64*embd);batch.embd_h = hidden.data();
        FILE * out = std::fopen(argv[3], "wb");if (!out) throw std::runtime_error("output");
        auto decode = [&](int start, int n, bool reuse) {
            common_batch_clear(batch);llama_set_mtp_index_reuse(ctx.get(), reuse);
            for (unsigned seq = 0; seq < cp.n_seq_max; ++seq) {
                for (int pos = start; pos < start + n; ++pos) {
                    int row = batch.n_tokens;
                    common_batch_add(batch, (pos*3 + seq + 1)%vocab, pos, {llama_seq_id(seq)}, pos + 1 == start + n);
                    for (int i = 0; i < embd; ++i) hidden[row*embd + i] = 0.01f*std::sin(float(pos*embd+i));
                }
            }
            if (llama_decode(ctx.get(), batch)) throw std::runtime_error("decode");
            for (int row = 0; row < batch.n_tokens; ++row) if (batch.logits[row]) {
                const float * logits = llama_get_logits_ith(ctx.get(), row);
                if (std::fwrite(logits, sizeof(float), vocab, out) != size_t(vocab)) throw std::runtime_error("write");
            }
            printf("{\"pos\":%d,\"requested_reuse\":%s,\"score\":%d,\"key\":%d,\"gate\":%d}\n",
                    start,reuse?"true":"false",counts.score,counts.key,counts.gate);fflush(stdout);
        };
        decode(0,16,false);decode(16,1,false);decode(17,1,true);decode(18,1,true);
        std::fclose(out);batch.embd_h = nullptr;llama_batch_free(batch);
        printf("{\"done\":true,\"rows\":%u,\"vocab\":%d}\n",4*cp.n_seq_max,vocab);
        return 0;
    } catch (const std::exception & e) { fprintf(stderr,"%s\n",e.what());return 2; }
}
