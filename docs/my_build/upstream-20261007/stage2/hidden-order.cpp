#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}
static void compare(const std::vector<float> & a, const std::vector<float> & b, const char * label, bool exact = false) {
    require(a.size() == b.size(), "size mismatch");
    double diff = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite value");
        diff += double(a[i] - b[i]) * (a[i] - b[i]);
        norm += double(b[i]) * b[i];
    }
    if (exact ? a != b : diff > 1e-5 * std::max(norm, 1e-20)) {
        fprintf(stderr, "%s NMSE %.9g\n", label, diff / std::max(norm, 1e-20));
        throw std::runtime_error(label);
    }
}
struct snapshot { std::vector<float> layer, nextn, logits; };
static snapshot read(llama_context * ctx, llama_model * model, const std::vector<int> & order,
                     int mask, int nextn_mode, bool capture, bool tensor_first) {
    const int embd = llama_model_n_embd(model), hidden = llama_model_n_embd_out(model);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    snapshot result;
    result.layer.resize(capture ? 15 * embd : 0);
    result.nextn.resize(nextn_mode ? 15 * hidden : 0);
    result.logits.resize(15 * vocab);
    ggml_tensor * tensor = nullptr;
    if (nextn_mode >= 3 && tensor_first) {
        tensor = llama_get_embeddings_nextn_tensor(ctx);
    }
    const float * layer = capture ? llama_get_embeddings_layer_inp(ctx, 1) : nullptr;
    if (capture) { require(layer != nullptr, "no layer input"); }
    std::vector<float> device;
    if (nextn_mode >= 3) {
        if (!tensor) { tensor = llama_get_embeddings_nextn_tensor(ctx); }
        require(tensor != nullptr, "no backend NextN");
        device.resize(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, device.data(), 0, ggml_nbytes(tensor));
    }
    const bool masked = nextn_mode == 2 || nextn_mode == 4;
    const float * host = nextn_mode && nextn_mode < 3 ? llama_get_embeddings_nextn(ctx) : nullptr;
    if (nextn_mode && nextn_mode < 3) { require(host != nullptr, "no host NextN"); }
    size_t output_row = 0;
    for (size_t row = 0; row < order.size(); ++row) {
        const int logical = order[row], pos = logical % 5;
        const bool output = mask == 2 || pos == 4 || (mask == 1 && pos == 1);
        if (capture) {
            std::copy_n(layer + row * embd, embd, result.layer.data() + logical * embd);
        }
        if (nextn_mode && (!masked || output)) {
            const size_t nextn_row = masked ? output_row : row;
            const float * data = nextn_mode >= 3 ? device.data() : host;
            std::copy_n(data + nextn_row * hidden, hidden, result.nextn.data() + logical * hidden);
            if (nextn_mode < 3) {
                const float * ith = llama_get_embeddings_nextn_ith(ctx, row);
                require(ith && std::memcmp(ith, data + nextn_row * hidden, hidden * sizeof(float)) == 0,
                        "NextN ith getter mismatch");
            }
        }
        if (output) {
            const float * logits = llama_get_logits_ith(ctx, row);
            require(logits != nullptr, "no logits");
            std::copy_n(logits, vocab, result.logits.data() + logical * vocab);
            ++output_row;
        }
    }
    return result;
}
static snapshot run(llama_context * ctx, llama_model * model, bool interleaved, int mask,
                    int nextn_mode, bool capture, bool tensor_first) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_embeddings_layer_inp(ctx, 1, capture);
    llama_set_embeddings_nextn(ctx, nextn_mode != 0, nextn_mode == 2 || nextn_mode == 4);
    llama_set_embeddings_nextn_host(ctx, nextn_mode < 3);
    llama_batch batch = llama_batch_init(15, 0, 1);
    std::vector<int> order;
    for (int i = 0; i < 15; ++i) {
        const int seq = interleaved ? i % 3 : i / 5;
        const int pos = interleaved ? i / 3 : i % 5;
        order.push_back(seq * 5 + pos);
        batch.token[i] = 1 + seq * 23 + pos * 3;
        batch.pos[i] = pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = mask == 2 || pos == 4 || (mask == 1 && pos == 1);
    }
    batch.n_tokens = 15;
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(status == 0, "decode failed");
    auto first = read(ctx, model, order, mask, nextn_mode, capture, tensor_first);
    auto second = read(ctx, model, order, mask, nextn_mode, capture, !tensor_first);
    compare(first.layer, second.layer, "repeated layer getter", true);
    compare(first.nextn, second.nextn, "repeated NextN getter", true);
    compare(first.logits, second.logits, "repeated logits getter", true);
    return first;
}
int main(int argc, char ** argv) {
    try {
        require(argc >= 2, "need model directory");
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level == GGML_LOG_LEVEL_ERROR) { fputs(text, stderr); }
        }, nullptr);
        ggml_backend_load_all();
        struct placement { const char * name; int gpu; llama_split_mode split; };
        const placement placements[] = {{"cpu", -1, LLAMA_SPLIT_MODE_LAYER}, {"gpu0", 0, LLAMA_SPLIT_MODE_LAYER},
            {"gpu1", 1, LLAMA_SPLIT_MODE_LAYER}, {"layer", 2, LLAMA_SPLIT_MODE_LAYER},
            {"row", 2, LLAMA_SPLIT_MODE_ROW}, {"tensor", 2, LLAMA_SPLIT_MODE_TENSOR}};
        std::vector<ggml_backend_dev_t> gpu;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpu.push_back(dev); }
        }
        size_t passed = 0, skipped = 0;
        for (const char * fixture : {"llama-dense", "glm5next-moe", "qwen4exp-moe"}) {
            const bool llama = std::string(fixture) == "llama-dense";
            const bool qwen = std::string(fixture) == "qwen4exp-moe";
            if (argc >= 3 && std::string(argv[2]) != fixture) { continue; }
            for (auto placement : placements) {
                if (std::getenv("ONLY_CPU") && placement.gpu != -1) { continue; }
                if ((placement.gpu >= 0 && gpu.size() < size_t(placement.gpu == 2 ? 2 : placement.gpu + 1)) ||
                    (!llama && placement.split == LLAMA_SPLIT_MODE_TENSOR)) { ++skipped; continue; }
                std::vector<ggml_backend_dev_t> devices;
                if (placement.gpu == 2) { devices = {gpu[0], gpu[1]}; }
                else if (placement.gpu >= 0) { devices = {gpu[placement.gpu]}; }
                devices.push_back(nullptr);
                auto mp = llama_model_default_params();
                mp.devices = devices.data();
                mp.n_gpu_layers = placement.gpu >= 0 ? 99 : 0;
                mp.split_mode = placement.split;
                if (placement.split == LLAMA_SPLIT_MODE_ROW) {
                    bool supported = true;
                    for (auto dev : devices) {
                        if (dev && !ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_split_buffer_type")) { supported = false; }
                    }
                    if (!supported) { printf("SKIP %s row: split buffers unavailable\n", fixture); ++skipped; continue; }
                }
                mp.load_mode = LLAMA_LOAD_MODE_NONE;
                auto path = std::string(argv[1]) + "/" + fixture + ".gguf";
                llama_model * model = llama_model_load_from_file(path.c_str(), mp);
                require(model != nullptr, "load failed");
                for (bool unified : {false, true}) {
                    for (unsigned ubatch : {6u, 32u}) {
                        auto cp = llama_context_default_params();
                        cp.n_ctx = 256; cp.n_batch = 32; cp.n_ubatch = ubatch; cp.n_seq_max = 3;
                        cp.n_threads = cp.n_threads_batch = 2; cp.kv_unified = unified;
                        cp.flash_attn_type = placement.split == LLAMA_SPLIT_MODE_TENSOR ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
                        llama_context * ctx = llama_init_from_model(model, cp);
                        require(ctx != nullptr, "context failed");
                        for (int mode = llama ? 0 : 1; mode <= (llama ? 0 : 4); ++mode) {
                            if (argc >= 4 && mode != std::stoi(argv[3])) { continue; }
                            for (bool capture : {false, true}) {
                                if ((llama && !capture) || (qwen && capture)) { continue; }
                                for (int mask = 0; mask < 3; ++mask) {
                                    auto reference = run(ctx, model, false, mask, mode, capture, true);
                                    if (const char * directory = std::getenv("DUMP_REFERENCE_PATH")) {
                                        const auto name = std::string(directory) + "/" + fixture + "-" + placement.name + "-" +
                                            std::to_string(unified) + "-" + std::to_string(ubatch) + "-" + std::to_string(mode) + "-" +
                                            std::to_string(capture) + "-" + std::to_string(mask) + ".bin";
                                        FILE * file = fopen(name.c_str(), "wb"); require(file, "dump open failed");
                                        for (const auto * data : {&reference.layer, &reference.nextn, &reference.logits}) {
                                            require(fwrite(data->data(), sizeof(float), data->size(), file) == data->size(), "dump failed");
                                        }
                                        fclose(file);
                                        ++passed;
                                        continue;
                                    }
                                    for (bool first : {false, true}) {
                                        auto actual = run(ctx, model, true, mask, mode, capture, first);
                                        compare(actual.layer, reference.layer, "original layer row order");
                                        compare(actual.nextn, reference.nextn, "original NextN row order");
                                        compare(actual.logits, reference.logits, "original logits row order");
                                    }
                                    ++passed;
                                }
                            }
                        }
                        llama_free(ctx);
                    }
                }
                llama_model_free(model);
                printf("PASS %s %s total=%zu\n", fixture, placement.name, passed); fflush(stdout);
            }
        }
        printf("RESULT passed=%zu skipped_placements=%zu\n", passed, skipped);
        llama_backend_free();
    } catch (const std::exception & error) { fprintf(stderr, "FAIL %s\n", error.what()); return 1; }
}
