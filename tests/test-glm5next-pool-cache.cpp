#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void set_pool_cache(bool enabled) {
#ifdef _WIN32
    _putenv_s("LLAMA_GLM5_POOL_CACHE", enabled ? "1" : "0");
#else
    setenv("LLAMA_GLM5_POOL_CACHE", enabled ? "1" : "0", 1);
#endif
}

static void log_errors(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static std::vector<float> run_decode(
        llama_model * model,
        std::vector<llama_token> & tokens,
        bool pool_cache) {
    set_pool_cache(pool_cache);

    llama_context_params params = llama_context_default_params();
    params.n_ctx = tokens.size() + 1;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_seq_max = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = GGML_TYPE_Q8_0;
    params.type_v = GGML_TYPE_Q8_0;

    llama_context * ctx = llama_init_from_model(model, params);
    if (ctx == nullptr) {
        return {};
    }

    constexpr size_t n_prefill = 4096;
    for (size_t off = 0; off < n_prefill; off += params.n_batch) {
        const size_t n = std::min<size_t>(params.n_batch, n_prefill - off);
        llama_batch batch = llama_batch_get_one(tokens.data() + off, n);
        if (llama_decode(ctx, batch) != 0) {
            llama_free(ctx);
            return {};
        }
    }

    for (size_t off = n_prefill; off < tokens.size(); ++off) {
        llama_batch batch = llama_batch_get_one(tokens.data() + off, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_free(ctx);
            return {};
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    std::vector<float> result(logits, logits + n_vocab);

    llama_free(ctx);
    return result;
}

static bool decode_sequence(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t offset,
        size_t count,
        llama_seq_id seq_id) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;

    for (size_t i = 0; i < count; ++i) {
        batch.token[i] = tokens[offset + i];
        batch.pos[i] = offset + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i] = i + 1 == count;
    }

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool run_multistream_restore(
        llama_model * model,
        const std::vector<llama_token> & tokens) {
    constexpr size_t n_prefill = 4096;
    constexpr size_t n_batch = 512;

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 2*(n_prefill + 1);
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_seq_max = 2;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = GGML_TYPE_Q8_0;
    params.type_v = GGML_TYPE_Q8_0;

    llama_context * ctx = llama_init_from_model(model, params);
    if (ctx == nullptr) {
        return false;
    }

    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        for (size_t off = 0; off < n_prefill; off += n_batch) {
            const size_t count = std::min(n_batch, n_prefill - off);
            if (!decode_sequence(ctx, tokens, off, count, seq_id)) {
                llama_free(ctx);
                return false;
            }
        }
    }

    constexpr llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t state_size = llama_state_seq_get_size_ext(ctx, 0, flags);
    std::vector<uint8_t> state(state_size);
    if (llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) != state.size() ||
        llama_state_seq_set_data_ext(ctx, state.data(), state.size(), 0, flags) != state.size()) {
        llama_free(ctx);
        return false;
    }

    // Restoring sequence 0 invalidates every pool map. Rebuilding it must not
    // clear the pending rebuild state of sequence 1's physical KV stream.
    const bool ok = decode_sequence(ctx, tokens, n_prefill, 1, 0) &&
                    decode_sequence(ctx, tokens, n_prefill, 1, 1);
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <glm5next.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    llama_log_set(log_errors, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;

    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        return 2;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        token = 1;
    }
    // Four decode steps cross a kpool=4 boundary and exercise both cache hits and
    // the insertion of a newly completed pool.
    std::vector<llama_token> tokens(4100, token);

    const std::vector<float> dense  = run_decode(model, tokens, false);
    const std::vector<float> cached = run_decode(model, tokens, true);

    if (dense.empty() || dense.size() != cached.size()) {
        llama_model_free(model);
        return 1;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < dense.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(dense[i] - cached[i]));
    }

    const size_t argmax_dense = std::max_element(dense.begin(), dense.end()) - dense.begin();
    const size_t argmax_cached = std::max_element(cached.begin(), cached.end()) - cached.begin();

    printf("pool cache: logits max abs = %.9g, argmax = %zu/%zu\n",
            (double) max_abs, argmax_dense, argmax_cached);

    const bool multistream_restore = run_multistream_restore(model, tokens);
    printf("pool cache: multi-stream restore = %s\n", multistream_restore ? "ok" : "failed");

    llama_model_free(model);

    return max_abs <= 2e-3f && argmax_dense == argmax_cached && multistream_restore ? 0 : 1;
}
