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

    llama_model_free(model);

    if (dense.empty() || dense.size() != cached.size()) {
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

    return max_abs <= 2e-3f && argmax_dense == argmax_cached ? 0 : 1;
}
