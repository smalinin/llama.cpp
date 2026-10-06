#include "llama-mmap.h"
#include "llama.h"
#include "sampling.h"
#include "chat-peg-parser.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    assert(argc == 3);
    const std::string mode = argv[1];
    if (mode == "dio") {
        llama_file file(argv[2], "rb", true);
        assert(file.has_direct_io());
        const std::vector<std::pair<size_t, size_t>> cases = {
            {1, 1}, {4095, 8195}, {4097, 64*1024*1024 - 17},
            {37, 64*1024*1024 + 12345}, {123, 192*1024*1024 + 17}
        };
        for (auto [offset, size] : cases) {
            std::vector<unsigned char> data(size);
            file.seek(offset, SEEK_SET);
            file.read_raw(data.data(), size);
            for (size_t i = 0; i < size; ++i) {
                assert(data[i] == (unsigned char) ((offset + i) % 251));
            }
            printf("DIO offset=%zu size=%zu OK\n", offset, size);
        }
    } else if (mode == "tool") {
        common_chat_msg msg;
        common_chat_peg_mapper mapper(msg);
        auto map = [&](const std::string & tag, const std::string & text) {
            mapper.map({0, "", tag, 0, text.size(), text, {}});
        };
        map(common_chat_peg_builder::TOOL_OPEN, "");
        map(common_chat_peg_builder::TOOL_ID, std::string(256, 'a'));
        map(common_chat_peg_builder::TOOL_CLOSE, "");
        map(common_chat_peg_builder::TOOL_ID, std::string(256, 'b'));
        map(common_chat_peg_builder::TOOL_OPEN, "");
        map(common_chat_peg_builder::TOOL_NAME, "next_tool");
        map(common_chat_peg_builder::TOOL_ARGS, "{}");
        map(common_chat_peg_builder::TOOL_CLOSE, "");
        assert(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "next_tool");
        puts("TOOL_CLOSE then TOOL_ID and next call OK");
    } else if (mode == "eog") {
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[2], mp);
        assert(model);
        auto cp = llama_context_default_params();
        cp.n_ctx = 64; cp.n_batch = 8; cp.n_ubatch = 8;
        auto * ctx = llama_init_from_model(model, cp);
        assert(ctx);
        const auto * vocab = llama_model_get_vocab(model);
        const auto eog = llama_vocab_eos(vocab);
        assert(eog >= 0 && llama_vocab_is_eog(vocab, eog));
        llama_token plain = 0;
        while (llama_vocab_is_eog(vocab, plain)) { ++plain; }
        auto batch = llama_batch_init(4, 0, 1);
        batch.n_tokens = 4;
        for (int i = 0; i < 4; ++i) {
            batch.token[i] = plain; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
        }
        assert(llama_decode(ctx, batch) == 0);
        for (int i = 0; i < 4; ++i) {
            auto * logits = llama_get_logits_ith(ctx, i);
            std::fill(logits, logits + llama_vocab_n_tokens(vocab), -1000.0f);
            logits[i == 0 ? eog : plain] = 1000.0f;
        }
        common_params_sampling sp;
        sp.temp = 0;
        auto * sampler = common_sampler_init(model, sp);
        auto accepted = common_sampler_sample_and_accept_n(sampler, ctx, {eog, plain, plain});
        printf("EOG interior accepted=%zu expected=1\n", accepted.size());
        fflush(stdout);
        assert(accepted.size() == 1 && accepted[0] == eog);
        common_sampler_free(sampler);
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    } else {
        return 2;
    }
}
