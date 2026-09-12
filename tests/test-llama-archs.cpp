#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct file_deleter {
    void operator()(FILE * file) const {
        if (file != nullptr) {
            fclose(file);
        }
    }
};

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void test_hybrid_memory_status() {
    GGML_ASSERT(llama_memory_status_combine(
        LLAMA_MEMORY_STATUS_SUCCESS,
        LLAMA_MEMORY_STATUS_NO_UPDATE,
        LLAMA_MEMORY_STATUS_FAILED_PREPARE) == LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    GGML_ASSERT(llama_memory_status_combine(
        LLAMA_MEMORY_STATUS_NO_UPDATE,
        LLAMA_MEMORY_STATUS_SUCCESS,
        LLAMA_MEMORY_STATUS_FAILED_COMPUTE) == LLAMA_MEMORY_STATUS_FAILED_COMPUTE);
    GGML_ASSERT(llama_memory_status_combine(
        LLAMA_MEMORY_STATUS_NO_UPDATE,
        LLAMA_MEMORY_STATUS_NO_UPDATE,
        LLAMA_MEMORY_STATUS_NO_UPDATE) == LLAMA_MEMORY_STATUS_NO_UPDATE);
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v N] [-h/--help]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool mtp = false) {
    GGML_ASSERT(!mtp || arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_QWEN4EXP);
    gguf_context_ptr  ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        n_embd  = 32;
        n_head  = 4;
        n_ff    = 32;
        n_layer = 2;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_QWEN4EXP && mtp) {
        n_layer = 3;  // two trunk layers and one NextN/MTP layer
    } else if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA ||
               arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 ||
               arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_HY_V4 ||
               arch == LLM_ARCH_GLM5NEXT) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_GLM5NEXT && mtp) {
            n_layer = 3; // two trunk layers and one NextN/MTP layer
        }
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        n_head_kv = 1;
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,             uint32_t(1));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_INDEX_SHARE_MTP, true);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        // head_count doubles as the KDA head count, so it stays uniform; the kv array is what
        // marks the recurrent layers, and the loader asserts it holds both a zero and a nonzero
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il == 1 ? 0 : n_head_kv);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH, uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, uint32_t(8));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, uint32_t(2));
    } else if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA ||
               arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 ||
               arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        // nope-only MLA: the cache holds the bare latent, so no rope width is added on top of
        // the kv LoRA rank and n_rot has to be an explicit 0, not the head size default
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(512));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    // glm5next warns on anything but the 1e-6 its indexer k_norm hardcodes
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          arch == LLM_ARCH_GLM5NEXT ? 1e-6f : 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK, arch == LLM_ARCH_DEEPSEEK4  ? uint32_t(64) :
                                            arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(16) :
                                                                          uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        if (!mtp) {
            // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
            const uint32_t ple_ngram_size      = 3;
            const uint32_t ple_heads_per_ngram = 2;
            const uint32_t ple_n_heads         = (ple_ngram_size - 1) * ple_heads_per_ngram;
            GGML_ASSERT(n_embd % ple_n_heads == 0);
            const uint32_t ple_head_dim = n_embd / ple_n_heads;

            std::vector<uint64_t> ple_head_offsets(ple_n_heads);
            std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
            for (uint32_t h = 0; h < ple_n_heads; h++) {
                ple_head_offsets[h] = uint64_t(h) * n_vocab;
            }

            // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
            ms.add_kv(LLM_KV_PLE_LAYERS, std::vector<uint32_t>({ 0 }));
            ms.add_kv(LLM_KV_PLE_NGRAM_SIZE, ple_ngram_size);
            ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM, ple_heads_per_ngram);
            ms.add_kv(LLM_KV_PLE_CONV_KERNEL, uint32_t(4));
            ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID, uint32_t(0));
            ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, ple_head_dim);
            ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS, std::vector<uint64_t>({ 1, 3, 5 }));
            ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS, ple_head_offsets);
            ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES, ple_head_vocab_sizes);
        }
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(2) :
                                                   arch == LLM_ARCH_MINIMAX_M3 ? n_head :
                                                                                 uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(4) :
                                                   arch == LLM_ARCH_QWEN4EXP   ? n_embd_head :
                                                                                 uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(2) : uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK, uint32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>({ 1, 1 }));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, 160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON, 1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT, uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, std::vector<float>({ 10.0f, 10.0f }));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({ 10.0f, 10.0f }));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE, 1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER_ID, uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,      uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_TOP_K_BLOCKS,    uint32_t(1));

        ms.add_kv(LLM_KV_ENGRAM_HEAD_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_KEY_LENGTH, uint32_t(8));
        ms.add_kv(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_LAYER_IDS, std::vector<int32_t>({ 0 }));
        ms.add_kv(LLM_KV_ENGRAM_MULTIPLIERS, std::vector<uint64_t>({ 3, 5 }));
        ms.add_kv(LLM_KV_ENGRAM_PRIMES, std::vector<uint64_t>({ 67, 71 }));
        ms.add_kv(LLM_KV_ENGRAM_OFFSETS, std::vector<uint64_t>({ 0, 67 }));
        ms.add_kv(LLM_KV_ENGRAM_PAD_ID, uint32_t(2));

        std::vector<int32_t> token_map(n_vocab);
        std::iota(token_map.begin(), token_map.end(), 0);
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP, token_map);
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4)); // build_hc_pre asserts exactly 4 streams
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        // the only arch that pools indexer keys; top_k must be a whole number of pools, and
        // the resulting selection width has to stay under n_ctx or the sparse path goes unused
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,              uint32_t(4));
        // glm5next reads these unconditionally; the if (moe) block below never sets them
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);
    }
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,
                  arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41 ? uint32_t(4) :
                                                                              uint32_t(2));  // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }

    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
    struct gguf_context *                   gguf_ctx,
    FILE *                                  file,
    const size_t                            seed,
    const std::vector<ggml_backend_dev_t> & devs,
    const llama_split_mode                  split_mode        = LLAMA_SPLIT_MODE_LAYER,
    bool                                    encode            = false,
    const llama_context_type                ctx_type          = LLAMA_CONTEXT_TYPE_DEFAULT,
    ggml_backend_sched_eval_callback        cb_eval           = nullptr,
    void *                                  cb_eval_user_data = nullptr,
    uint32_t                                n_seq_max         = 1,
    llama_context *                         ctx_other         = nullptr,
    ggml_type                               type_k            = GGML_TYPE_F16) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = ctx_type == LLAMA_CONTEXT_TYPE_MTP;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.ctx_type             = ctx_type;
    ctx_params.n_ctx                = 0;
    ctx_params.n_threads            = 4;
    ctx_params.n_threads_batch      = 4;
    ctx_params.n_seq_max            = n_seq_max;
    ctx_params.cb_eval              = cb_eval;
    ctx_params.cb_eval_user_data    = cb_eval_user_data;
    ctx_params.ctx_other            = ctx_other;
    ctx_params.type_k               = type_k;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static void save_qwen4exp_shared_mtp(const llama_model * model, FILE * file) {
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    GGML_ASSERT(gguf_remove_key(ms.gguf_ctx, "qwen4exp.attention.recurrent_layers") >= 0);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    const std::string block_prefix = "blk." + std::to_string(llama_model_n_layer(model)) + ".";
    const std::string token_embd   = block_prefix + "nextn.embed_tokens.";
    const std::string output       = block_prefix + "nextn.shared_head_head.";
    for (const auto & [name, tensor] : model->tensors_by_name) {
        if (name.compare(0, block_prefix.size(), block_prefix) == 0 &&
            name.compare(0, token_embd.size(), token_embd) != 0 && name.compare(0, output.size(), output) != 0) {
            ms.add_tensor(tensor);
        }
    }
    ms.save(file);
    rewind(file);
}

static void check_qwen4exp_shared_mtp(const llama_model * model) {
    const uint32_t il = llama_model_n_layer(model);
    GGML_ASSERT(model->tok_embd == nullptr && model->output == nullptr);
    GGML_ASSERT(model->layers[il].nextn.embed_tokens == nullptr);
    GGML_ASSERT(model->layers[il].nextn.shared_head_head == nullptr);
}

struct mtp_indexer_eval_count {
    int score        = 0;
    int key          = 0;
    int gate         = 0;
    int reuse        = 0;
    int device_write = 0;
    int cache_rot    = 0;
    int cache_unrot  = 0;
};

static void set_mtp_topk_share_env(bool enabled) {
#ifdef _WIN32
    _putenv_s("LLAMA_GLM5_MTP_TOPK_SHARE", enabled ? "1" : "0");
#else
    setenv("LLAMA_GLM5_MTP_TOPK_SHARE", enabled ? "1" : "0", 1);
#endif
}

static void set_mtp_device_draft_env(bool enabled) {
#ifdef _WIN32
    _putenv_s("LLAMA_MTP_DEVICE_DRAFT", enabled ? "1" : "0");
#else
    setenv("LLAMA_MTP_DEVICE_DRAFT", enabled ? "1" : "0", 1);
#endif
}

static bool count_mtp_indexer_score(ggml_tensor * tensor, bool ask, void * user_data) {
    static const char score_pool_name[]  = "indexer_pool_score";
    static const char score_block_name[] = "indexer_score_blk";
    static const char key_name[]         = "indexer_k";
    static const char gate_name[]        = "indexer_gate";
    static const char reuse_name[]       = "indexer_top_k_blk_reused";
    static const char device_name[]      = "mtp_device_write";
    static const char cache_rot_name[]   = "indexer_k_cache_rot";
    static const char cache_unrot_name[] = "indexer_k_cache_unrot";
    const bool        score              = strncmp(tensor->name, score_pool_name, strlen(score_pool_name)) == 0 ||
                       strncmp(tensor->name, score_block_name, strlen(score_block_name)) == 0;
    const bool key    = strncmp(tensor->name, key_name, strlen(key_name)) == 0;
    const bool gate   = strncmp(tensor->name, gate_name, strlen(gate_name)) == 0;
    const bool reuse  = strncmp(tensor->name, reuse_name, strlen(reuse_name)) == 0;
    const bool device = strncmp(tensor->name, device_name, strlen(device_name)) == 0;
    const bool cache_rot   = strncmp(tensor->name, cache_rot_name, strlen(cache_rot_name)) == 0;
    const bool cache_unrot = strncmp(tensor->name, cache_unrot_name, strlen(cache_unrot_name)) == 0;
    if (ask) {
        return score || key || gate || reuse || device || cache_rot || cache_unrot;
    }
    auto * count = static_cast<mtp_indexer_eval_count *>(user_data);
    if (score) {
        count->score++;
    }
    if (key) {
        count->key++;
    }
    if (gate) {
        count->gate++;
    }
    if (reuse) {
        count->reuse++;
    }
    if (device) {
        count->device_write++;
    }
    if (cache_rot) {
        count->cache_rot++;
    }
    if (cache_unrot) {
        count->cache_unrot++;
    }
    return true;
}

struct mtp_draft_result {
    std::vector<float> logits;
    std::vector<llama_token> tokens;
    std::vector<float> probs;
};

static mtp_draft_result get_mtp_draft(llama_model *                    model,
                                      llama_context *                  lctx,
                                      const std::vector<llama_token> & tokens,
                                      const std::vector<llama_token> * reference_tokens = nullptr,
                                      mtp_indexer_eval_count *         eval_count       = nullptr,
                                      bool                             prefill          = true,
                                      bool                             expect_cache_rot = false) {
    static constexpr uint32_t n_prefix = 16;
    static constexpr uint32_t n_steps  = 2;

    const uint32_t n_vocab       = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd        = llama_model_n_embd_out(model);
    const uint32_t n_ctx         = llama_n_ctx(lctx);
    const uint32_t n_seq         = llama_n_seq_max(lctx);
    char           arch_name[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_name, sizeof(arch_name));
    const bool         qwen4exp = strcmp(arch_name, llm_arch_name(LLM_ARCH_QWEN4EXP)) == 0;
    llama_batch        batch    = llama_batch_init(n_ctx, 0, 1);
    std::vector<float> batch_h((size_t) n_ctx * n_embd);
    batch.embd_h = batch_h.data();

    GGML_ASSERT(tokens.size() > n_prefix);
    GGML_ASSERT((n_prefix + n_steps) * n_seq <= n_ctx);
    GGML_ASSERT(reference_tokens == nullptr || reference_tokens->size() == n_steps * n_seq);

    auto set_synthetic_h = [&](uint32_t pos, uint32_t row) {
        for (uint32_t i = 0; i < n_embd; ++i) {
            batch.embd_h[(size_t) row*n_embd + i] = 0.01f*sinf(float(pos*n_embd + i));
        }
    };

    llama_set_embeddings_nextn(lctx, true, true);
    llama_set_mtp_index_reuse(lctx, false);
    if (prefill) {
        for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
            for (uint32_t pos = 0; pos < n_prefix; ++pos) {
                common_batch_add(batch, tokens[pos], pos, { (llama_seq_id) seq_id }, pos + 1 == n_prefix);
                set_synthetic_h(pos, batch.n_tokens - 1);
            }
        }
        if (llama_decode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to prefill MTP cache");
        }
    }

    mtp_draft_result result;
    result.logits.reserve(n_steps * n_seq * n_vocab);
    result.tokens.reserve(n_steps * n_seq);

    std::vector<float> h((size_t) n_seq * n_embd);
    for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
        for (uint32_t i = 0; i < n_embd; ++i) {
            h[(size_t) seq_id * n_embd + i] = 0.01f * sinf(float(n_prefix * n_embd + i));
        }
    }

    std::vector<llama_token> next_tokens(n_seq, tokens[n_prefix]);
    for (uint32_t step = 0; step < n_steps; ++step) {
        common_batch_clear(batch);

        for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
            const size_t      idx   = (size_t) step * n_seq + seq_id;
            const llama_token token = reference_tokens ? (*reference_tokens)[idx] : next_tokens[seq_id];
            result.tokens.push_back(token);
            common_batch_add(batch, token, n_prefix + step, { (llama_seq_id) seq_id }, true);
            memcpy(batch.embd_h + (size_t) (batch.n_tokens - 1) * n_embd, h.data() + (size_t) seq_id * n_embd,
                   n_embd * sizeof(float));
        }

        llama_set_mtp_index_reuse(lctx, step > 0);
        if (eval_count) {
            eval_count->score = 0;
            eval_count->key   = 0;
            eval_count->gate  = 0;
            eval_count->reuse = 0;
            eval_count->cache_rot   = 0;
            eval_count->cache_unrot = 0;
        }

        if (llama_decode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode MTP draft step");
        }

        for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
            const float * logits = llama_get_logits_ith(lctx, seq_id);
            result.logits.insert(result.logits.end(), logits, logits + n_vocab);
        }

        if (eval_count && step == 0 && eval_count->score == 0) {
            llama_batch_free(batch);
            throw std::runtime_error("first MTP step did not score the indexer");
        }
        if (eval_count && step == 0 &&
            (eval_count->key == 0 || (!qwen4exp && eval_count->gate == 0) || eval_count->reuse != 0)) {
            llama_batch_free(batch);
            throw std::runtime_error("first MTP step did not compute a fresh indexer selection");
        }
        if (eval_count && step == 0 && expect_cache_rot &&
            (eval_count->cache_rot == 0 || eval_count->cache_unrot == 0)) {
            llama_batch_free(batch);
            throw std::runtime_error("quantized indexer cache rotation count: " +
                    std::to_string(eval_count->cache_rot) + "/" + std::to_string(eval_count->cache_unrot));
        }
        if (eval_count && step == 0 && qwen4exp && !expect_cache_rot &&
            (eval_count->cache_rot != 0 || eval_count->cache_unrot != 0)) {
            llama_batch_free(batch);
            throw std::runtime_error("unquantized indexer cache unexpectedly rotated keys");
        }
        if (eval_count && step > 0 && eval_count->score != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("reused MTP step recomputed indexer scores");
        }
        if (eval_count && step > 0 &&
            (eval_count->key != 0 || (!qwen4exp && eval_count->gate != 0) || (qwen4exp && eval_count->reuse == 0))) {
            llama_batch_free(batch);
            throw std::runtime_error("reused MTP step did not reuse the indexer selection");
        }

        if (step + 1 < n_steps) {
            for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
                const float * logits  = llama_get_logits_ith(lctx, seq_id);
                next_tokens[seq_id]   = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
                const float * h_nextn = llama_get_embeddings_nextn_ith(lctx, seq_id);
                memcpy(h.data() + (size_t) seq_id * n_embd, h_nextn, n_embd * sizeof(float));
            }
        }
    }

    llama_set_mtp_index_reuse(lctx, false);
    llama_batch_free(batch);
    return result;
}

static double check_mtp_parallel(const mtp_draft_result & reference,
                                 const mtp_draft_result & parallel,
                                 uint32_t                 n_vocab,
                                 uint32_t                 n_seq) {
    GGML_ASSERT(reference.logits.size() % n_vocab == 0);
    const size_t n_steps = reference.logits.size() / n_vocab;
    GGML_ASSERT(reference.tokens.size() == n_steps);
    GGML_ASSERT(parallel.tokens.size() == n_steps * n_seq);
    GGML_ASSERT(parallel.logits.size() == n_steps * n_seq * n_vocab);

    double max_nmse = 0.0;
    for (size_t step = 0; step < n_steps; ++step) {
        std::vector<float> ref_logits(reference.logits.begin() + step * n_vocab,
                                      reference.logits.begin() + (step + 1) * n_vocab);
        for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
            const size_t idx = step * n_seq + seq_id;
            GGML_ASSERT(parallel.tokens[idx] == reference.tokens[step]);
            std::vector<float> parallel_logits(parallel.logits.begin() + idx * n_vocab,
                                               parallel.logits.begin() + (idx + 1) * n_vocab);
            max_nmse = std::max(max_nmse, nmse(ref_logits, parallel_logits));
        }
    }
    return max_nmse;
}

static void check_mtp_rollback(llama_model *                    model,
                               llama_context *                  lctx,
                               const std::vector<llama_token> & tokens,
                               const mtp_draft_result &         reference,
                               mtp_indexer_eval_count *         eval_count) {
    static constexpr llama_pos n_prefix = 16;
    if (!llama_memory_seq_rm(llama_get_memory(lctx), 0, n_prefix, -1)) {
        throw std::runtime_error("failed to reject MTP draft tokens");
    }

    const auto replay = get_mtp_draft(model, lctx, tokens, &reference.tokens, eval_count, false);
    if (nmse(reference.logits, replay.logits) > 1e-12) {
        throw std::runtime_error("MTP draft rollback changed replay logits");
    }
}

static mtp_draft_result get_mtp_device_draft(llama_model *                    model,
                                             llama_context *                  lctx,
                                             const std::vector<llama_token> & tokens,
                                             mtp_indexer_eval_count *         eval_count = nullptr) {
    static constexpr uint32_t n_prefix = 16;
    static constexpr uint32_t n_steps  = 2;

    const uint32_t n_embd = llama_model_n_embd(model);
    const uint32_t n_ctx  = llama_n_ctx(lctx);
    const uint32_t n_seq  = llama_n_seq_max(lctx);
    GGML_ASSERT(n_seq > 1);
    GGML_ASSERT(n_prefix*n_seq + n_steps*n_seq <= n_ctx);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    std::vector<float> batch_h((size_t) n_ctx*n_embd);
    batch.embd_h = batch_h.data();

    auto set_synthetic_h = [&](uint32_t pos, uint32_t row) {
        for (uint32_t i = 0; i < n_embd; ++i) {
            batch.embd_h[(size_t) row*n_embd + i] = 0.01f*sinf(float(pos*n_embd + i));
        }
    };

    llama_set_embeddings_nextn(lctx, true, true);
    llama_set_embeddings_nextn_host(lctx, false);
    llama_set_mtp_index_reuse(lctx, false);
    for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
        for (uint32_t pos = 0; pos < n_prefix; ++pos) {
            common_batch_add(batch, tokens[pos], pos, {(llama_seq_id) seq_id}, pos + 1 == n_prefix);
            set_synthetic_h(pos, batch.n_tokens - 1);
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to prefill device-resident GLM5NEXT MTP cache");
    }

    std::vector<llama_sampler_ptr> backend_samplers;
    auto detach_samplers = [&]() {
        for (uint32_t seq_id = 0; seq_id < backend_samplers.size(); ++seq_id) {
            llama_set_sampler(lctx, seq_id, nullptr);
        }
    };
    auto fail = [&](const char * message) {
        detach_samplers();
        llama_batch_free(batch);
        throw std::runtime_error(message);
    };

    for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(10));
        if (!llama_set_sampler(lctx, seq_id, sampler.get())) {
            fail("failed to attach device-resident GLM5NEXT MTP sampler");
        }
        backend_samplers.push_back(std::move(sampler));
    }

    set_mtp_device_draft_env(false);
    if (llama_mtp_device_draft_begin(lctx, n_steps)) {
        fail("LLAMA_MTP_DEVICE_DRAFT=0 did not select the host fallback");
    }
    set_mtp_device_draft_env(true);
    if (!llama_mtp_device_draft_begin(lctx, n_steps)) {
        fail("failed to start device-resident GLM5NEXT MTP draft");
    }

    // Exercise the scheduler re-reserve performed by memory_update() while
    // device drafting is active. Context shift reaches this path after draft
    // setup; opposite shifts retain the reference cache positions and logits.
    auto * mem = llama_get_memory(lctx);
    for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
        llama_memory_seq_add(mem, seq_id, 0, -1,  1);
        llama_memory_seq_add(mem, seq_id, 0, -1, -1);
    }

    for (uint32_t step = 0; step < n_steps; ++step) {
        common_batch_clear(batch);
        for (uint32_t seq_id = 0; seq_id < n_seq; ++seq_id) {
            common_batch_add(batch, tokens[n_prefix], n_prefix + step, {(llama_seq_id) seq_id}, true);
            if (step == 0) {
                set_synthetic_h(n_prefix, batch.n_tokens - 1);
            }
        }

        llama_set_mtp_index_reuse(lctx, step > 0);
        if (eval_count) {
            eval_count->score        = 0;
            eval_count->key          = 0;
            eval_count->gate         = 0;
            eval_count->device_write = 0;
        }

        if (llama_decode(lctx, batch)) {
            fail("failed to decode device-resident GLM5NEXT MTP draft step");
        }
        if (eval_count && eval_count->device_write == 0) {
            fail("device-resident GLM5NEXT MTP draft did not update backend state");
        }
        if (eval_count && step == 0 &&
                (eval_count->score == 0 || eval_count->key == 0 || eval_count->gate == 0)) {
            fail("first device-resident GLM5NEXT MTP step did not compute the indexer");
        }
        if (eval_count && step > 0 &&
                (eval_count->score != 0 || eval_count->key != 0 || eval_count->gate != 0)) {
            fail("reused device-resident GLM5NEXT MTP step recomputed the indexer");
        }

        if (step + 1 < n_steps) {
            llama_mtp_device_draft_advance(lctx);
        }
    }

    mtp_draft_result result;
    result.tokens.resize(n_steps*n_seq);
    result.probs.resize(result.tokens.size());
    if (!llama_mtp_device_draft_finish(
            lctx, result.tokens.data(), result.probs.data(), result.tokens.size())) {
        fail("failed to finish device-resident GLM5NEXT MTP draft");
    }

    llama_set_mtp_index_reuse(lctx, false);
    detach_samplers();
    llama_batch_free(batch);
    return result;
}

static void check_mtp_device_draft(
        const mtp_draft_result & reference, const mtp_draft_result & device, uint32_t n_vocab) {
    GGML_ASSERT(device.tokens.size() == device.probs.size());
    GGML_ASSERT(reference.logits.size() % n_vocab == 0);
    const size_t n_steps = reference.logits.size()/n_vocab;
    GGML_ASSERT(device.tokens.size() % n_steps == 0);
    const size_t n_seq = device.tokens.size()/n_steps;

    for (size_t step = 0; step < n_steps; ++step) {
        const float * logits = reference.logits.data() + step*n_vocab;
        const llama_token expected = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);

        std::vector<float> top_logits(logits, logits + n_vocab);
        const size_t n_top = std::min<size_t>(10, top_logits.size());
        std::partial_sort(
                top_logits.begin(), top_logits.begin() + n_top, top_logits.end(), std::greater<float>());
        float sum = 0.0f;
        for (size_t i = 0; i < n_top; ++i) {
            sum += expf(top_logits[i] - top_logits[0]);
        }
        const float expected_p = 1.0f/sum;

        for (size_t seq_id = 0; seq_id < n_seq; ++seq_id) {
            const size_t result_idx = step*n_seq + seq_id;
            if (device.tokens[result_idx] != expected) {
                throw std::runtime_error("device-resident GLM5NEXT MTP selected token " +
                        std::to_string(device.tokens[result_idx]) + " instead of " +
                        std::to_string(expected) + " at step " + std::to_string(step));
            }
            if (fabsf(device.probs[result_idx] - expected_p) > 1e-5f) {
                throw std::runtime_error("device-resident GLM5NEXT MTP returned an unexpected confidence");
            }
        }
    }
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_GLM5NEXT:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr  gguf_ctx    = get_gguf_ctx(arch, moe);
            gguf_context_ptr  gguf_ctx_mtp =
                arch == LLM_ARCH_GLM5NEXT || arch == LLM_ARCH_QWEN4EXP ? get_gguf_ctx(arch, moe, true) : nullptr;
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float>                            logits_cpu;
            std::vector<float>                            logits_mtp_cpu;
            std::vector<llama_token>                      tokens_mtp_cpu;
            std::vector<float>                            logits_mtp_shared_cpu;
            std::vector<llama_token>                      tokens_mtp_shared_cpu;
            std::unique_ptr<FILE, file_deleter>           qwen_mtp_shared_file;

            if (arch == LLM_ARCH_QWEN4EXP) {
                mtp_indexer_eval_count eval_count;
                auto                   model_and_ctx_mtp_cpu =
                    get_model_and_ctx(gguf_ctx_mtp.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                                      LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &eval_count);
                auto draft_mtp_cpu = get_mtp_draft(model_and_ctx_mtp_cpu.first.get(),
                                                   model_and_ctx_mtp_cpu.second.get(), tokens, nullptr, &eval_count);
                check_mtp_rollback(model_and_ctx_mtp_cpu.first.get(), model_and_ctx_mtp_cpu.second.get(), tokens,
                                   draft_mtp_cpu, &eval_count);
                logits_mtp_cpu = std::move(draft_mtp_cpu.logits);
                tokens_mtp_cpu = std::move(draft_mtp_cpu.tokens);

                mtp_indexer_eval_count eval_count_q8;
                auto model_and_ctx_mtp_q8_cpu =
                    get_model_and_ctx(gguf_ctx_mtp.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                                      LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &eval_count_q8, 1, nullptr,
                                      GGML_TYPE_Q8_0);
                const char * attn_rot_disable = getenv("LLAMA_ATTN_ROT_DISABLE");
                const bool expect_cache_rot = attn_rot_disable == nullptr || atoi(attn_rot_disable) == 0;
                get_mtp_draft(model_and_ctx_mtp_q8_cpu.first.get(), model_and_ctx_mtp_q8_cpu.second.get(), tokens,
                              &tokens_mtp_cpu, &eval_count_q8, true, expect_cache_rot);

                qwen_mtp_shared_file.reset(tmpfile());
                if (qwen_mtp_shared_file) {
                    save_qwen4exp_shared_mtp(model_and_ctx_mtp_cpu.first.get(), qwen_mtp_shared_file.get());
                }
            }
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        double nmse_val = nmse(logits_cpu, logits_dev);
                        if (arch == LLM_ARCH_GLM5NEXT) {
                            if (logits_mtp_cpu.empty()) {
                                // The optimized graph must preserve the diagnostic fallback's
                                // logits while pruning indexer K/G after the first draft step.
                                set_mtp_topk_share_env(false);
                                auto model_and_ctx_mtp_fallback = get_model_and_ctx(
                                    gguf_ctx_mtp.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                                    LLAMA_CONTEXT_TYPE_MTP);
                                const auto draft_mtp_fallback = get_mtp_draft(
                                    model_and_ctx_mtp_fallback.first.get(), model_and_ctx_mtp_fallback.second.get(),
                                    tokens);

                                set_mtp_topk_share_env(true);
                                mtp_indexer_eval_count eval_count;
                                auto model_and_ctx_mtp_cpu = get_model_and_ctx(
                                    gguf_ctx_mtp.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                                    LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &eval_count);
                                auto draft_mtp_cpu = get_mtp_draft(
                                    model_and_ctx_mtp_cpu.first.get(), model_and_ctx_mtp_cpu.second.get(),
                                    tokens, &draft_mtp_fallback.tokens, &eval_count);
                                const uint32_t n_vocab_mtp = llama_vocab_n_tokens(
                                    llama_model_get_vocab(model_and_ctx_mtp_cpu.first.get()));
                                GGML_ASSERT(draft_mtp_fallback.logits.size() == draft_mtp_cpu.logits.size());
                                GGML_ASSERT(draft_mtp_cpu.logits.size() % n_vocab_mtp == 0);
                                for (size_t off = 0; off < draft_mtp_cpu.logits.size(); off += n_vocab_mtp) {
                                    const auto argmax_fallback = std::max_element(
                                        draft_mtp_fallback.logits.begin() + off,
                                        draft_mtp_fallback.logits.begin() + off + n_vocab_mtp);
                                    const auto argmax_reuse = std::max_element(
                                        draft_mtp_cpu.logits.begin() + off,
                                        draft_mtp_cpu.logits.begin() + off + n_vocab_mtp);
                                    if (argmax_fallback - draft_mtp_fallback.logits.begin() !=
                                            argmax_reuse - draft_mtp_cpu.logits.begin()) {
                                        throw std::runtime_error("GLM5NEXT MTP indexer reuse changed greedy output");
                                    }
                                }
                                logits_mtp_cpu = std::move(draft_mtp_cpu.logits);
                                tokens_mtp_cpu = std::move(draft_mtp_cpu.tokens);
                            }

                            mtp_indexer_eval_count eval_count;
                            auto model_and_ctx_mtp_dev = get_model_and_ctx(
                                gguf_ctx_mtp.get(), nullptr, seed, dc.devs, dc.split_mode, false,
                                LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &eval_count);
                            const auto draft_mtp_dev = get_mtp_draft(
                                model_and_ctx_mtp_dev.first.get(), model_and_ctx_mtp_dev.second.get(),
                                tokens, &tokens_mtp_cpu, &eval_count);
                            nmse_val = std::max(nmse_val, nmse(logits_mtp_cpu, draft_mtp_dev.logits));

                            mtp_indexer_eval_count device_eval_count;
                            auto model_and_ctx_mtp_device = get_model_and_ctx(
                                gguf_ctx_mtp.get(), nullptr, seed, dc.devs, dc.split_mode, false,
                                LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &device_eval_count, 2);
                            const auto device_draft = get_mtp_device_draft(
                                model_and_ctx_mtp_device.first.get(), model_and_ctx_mtp_device.second.get(),
                                tokens, &device_eval_count);
                            check_mtp_device_draft(
                                draft_mtp_dev, device_draft,
                                llama_vocab_n_tokens(llama_model_get_vocab(model_and_ctx_mtp_device.first.get())));
                        } else if (arch == LLM_ARCH_QWEN4EXP) {
                            mtp_indexer_eval_count eval_count;
                            auto                   model_and_ctx_mtp_dev =
                                get_model_and_ctx(gguf_ctx_mtp.get(), nullptr, seed, dc.devs, dc.split_mode, false,
                                                  LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &eval_count);
                            const auto draft_mtp_dev =
                                get_mtp_draft(model_and_ctx_mtp_dev.first.get(), model_and_ctx_mtp_dev.second.get(),
                                              tokens, &tokens_mtp_cpu, &eval_count);
                            nmse_val = std::max(nmse_val, nmse(logits_mtp_cpu, draft_mtp_dev.logits));

                            static constexpr uint32_t n_seq_mtp = 2;
                            std::vector<llama_token>  parallel_tokens;
                            parallel_tokens.reserve(tokens_mtp_cpu.size() * n_seq_mtp);
                            for (llama_token token : tokens_mtp_cpu) {
                                parallel_tokens.insert(parallel_tokens.end(), n_seq_mtp, token);
                            }

                            mtp_indexer_eval_count parallel_eval_count;
                            auto                   model_and_ctx_mtp_parallel = get_model_and_ctx(
                                gguf_ctx_mtp.get(), nullptr, seed, dc.devs, dc.split_mode, false,
                                LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &parallel_eval_count, n_seq_mtp);
                            const auto draft_mtp_parallel = get_mtp_draft(
                                model_and_ctx_mtp_parallel.first.get(), model_and_ctx_mtp_parallel.second.get(), tokens,
                                &parallel_tokens, &parallel_eval_count);
                            mtp_draft_result draft_mtp_cpu = { logits_mtp_cpu, tokens_mtp_cpu, {} };
                            nmse_val =
                                std::max(nmse_val, check_mtp_parallel(draft_mtp_cpu, draft_mtp_parallel,
                                                                      llama_vocab_n_tokens(llama_model_get_vocab(
                                                                          model_and_ctx_mtp_parallel.first.get())),
                                                                      n_seq_mtp));

                            if (qwen_mtp_shared_file) {
                                if (logits_mtp_shared_cpu.empty()) {
                                    rewind(qwen_mtp_shared_file.get());
                                    mtp_indexer_eval_count shared_eval_count;
                                    auto                   model_and_ctx_mtp_shared_cpu = get_model_and_ctx(
                                        nullptr, qwen_mtp_shared_file.get(), seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                                        LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score, &shared_eval_count, 1,
                                        model_and_ctx_cpu.second.get());
                                    check_qwen4exp_shared_mtp(model_and_ctx_mtp_shared_cpu.first.get());
                                    auto draft_mtp_shared_cpu = get_mtp_draft(model_and_ctx_mtp_shared_cpu.first.get(),
                                                                              model_and_ctx_mtp_shared_cpu.second.get(),
                                                                              tokens, nullptr, &shared_eval_count);
                                    logits_mtp_shared_cpu     = std::move(draft_mtp_shared_cpu.logits);
                                    tokens_mtp_shared_cpu     = std::move(draft_mtp_shared_cpu.tokens);
                                }

                                rewind(qwen_mtp_shared_file.get());
                                mtp_indexer_eval_count shared_eval_count;
                                auto                   model_and_ctx_mtp_shared_dev =
                                    get_model_and_ctx(nullptr, qwen_mtp_shared_file.get(), seed, dc.devs, dc.split_mode,
                                                      false, LLAMA_CONTEXT_TYPE_MTP, count_mtp_indexer_score,
                                                      &shared_eval_count, 1, model_and_ctx_dev.second.get());
                                check_qwen4exp_shared_mtp(model_and_ctx_mtp_shared_dev.first.get());
                                const auto draft_mtp_shared_dev = get_mtp_draft(
                                    model_and_ctx_mtp_shared_dev.first.get(), model_and_ctx_mtp_shared_dev.second.get(),
                                    tokens, &tokens_mtp_shared_cpu, &shared_eval_count);
                                nmse_val = std::max(nmse_val, nmse(logits_mtp_shared_cpu, draft_mtp_shared_dev.logits));
                            }
                        }
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    test_hybrid_memory_status();

    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    std::string out;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, verbosity, out);
        }
        return test_backends(arch, seed, verbosity);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
