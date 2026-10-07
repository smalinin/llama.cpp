#include "llama.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * text) { if (!value) throw std::runtime_error(text); }
struct chunk { ggml_tensor * tensor; size_t offset, size; };
struct writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    std::vector<chunk> tensors;
    void write(const void * data, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(data);
        bytes.insert(bytes.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        std::vector<uint8_t> data(size);
        ggml_backend_tensor_get(t, data.data(), offset, size);
        write(data.data(), size);
        tensors.push_back({t, offset, size});
    }
    size_t n_bytes() override { return bytes.size(); }
};
struct reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t pos = 0, count = 0, fail_at;
    std::vector<chunk> written;
    explicit reader(const std::vector<uint8_t> & data, size_t fault) : bytes(data), fail_at(fault) {}
    void read(void * data, size_t size) override {
        require(pos + size <= bytes.size(), "metadata truncated");
        memcpy(data, bytes.data() + pos, size);
        pos += size;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        if (count++ == fail_at) throw std::runtime_error("injected tensor-read failure");
        require(pos + size <= bytes.size(), "tensor truncated");
        std::vector<uint8_t> poison(size, 0xff);
        ggml_backend_tensor_set(t, poison.data(), offset, size);
        written.push_back({t, offset, size});
        pos += size;
    }
    size_t n_bytes() override { return pos; }
};
static std::vector<float> decode(llama_context * ctx, llama_seq_id seq, int start, int n) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = 1 + (start + i)*3;
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = i == n - 1;
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(status == 0, "decode failed");
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "no logits");
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    return std::vector<float>(logits, logits + nv);
}
static writer snapshot(llama_memory_t memory, llama_seq_id seq, llama_state_seq_flags flags=0) {
    writer data;
    memory->state_write(data, seq, flags);
    return data;
}
int main(int argc, char ** argv) {
    try {
        require(argc >= 2, "fixture directory is required");
        ggml_backend_load_all();
        std::vector<ggml_backend_dev_t> gpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) gpus.push_back(dev);
        }
        int passed = 0, skipped = 0;
        for (const char * fixture : {"llama-dense", "glm5next-moe", "qwen4exp-moe", "glm-dsa-moe", "deepseek41-moe"}) {
            if (argc >= 3 && std::string(fixture) != argv[2]) continue;
            for (int gpu = -1; gpu < (int)gpus.size(); ++gpu) {
                if (getenv("ONLY_CPU") && gpu >= 0) continue;
                auto mp = llama_model_default_params();
                std::vector<ggml_backend_dev_t> devs;
                if (gpu >= 0) { devs = {gpus[gpu], nullptr}; mp.devices = devs.data(); }
                mp.n_gpu_layers = gpu >= 0 ? 99 : 0;
                auto path = std::string(argv[1]) + "/" + fixture + ".gguf";
                llama_model_ptr model(llama_model_load_from_file(path.c_str(), mp));
                require(model != nullptr, "model load failed");
                for (bool unified : {false, true}) {
                    for (ggml_type kv : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
                        auto cp = llama_context_default_params();
                        cp.n_ctx = 256; cp.n_batch = 32; cp.n_ubatch = 8; cp.n_seq_max = 3;
                        cp.n_threads = cp.n_threads_batch = 2; cp.kv_unified = unified;
                        cp.type_k = cp.type_v = kv;
                        cp.flash_attn_type = kv == GGML_TYPE_F16 ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
                        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
                        if (!ctx) { ++skipped; printf("SKIP %s gpu=%d unified=%d kv=%s\n",fixture,gpu,unified,ggml_type_name(kv)); continue; }
                        auto * mem = llama_get_memory(ctx.get());
                        for (int fault_mode = 0; fault_mode < 2; ++fault_mode) {
                            llama_memory_clear(mem, true);
                            decode(ctx.get(), 0, 0, 16);
                            decode(ctx.get(), 1, 0, 8);
                            auto preserved = snapshot(mem, 1);
                            auto source = snapshot(mem, 0);
                            require(!source.tensors.empty(), "no serialized tensors");
                            require(llama_memory_seq_rm(mem, 0, -1, -1), "remove failed");
                            const size_t fault = fault_mode == 0 ? source.tensors.size()/2 : source.tensors.size() - 1;
                            reader io(source.bytes, fault);
                            bool failed = false;
                            try { mem->state_read(io, 0); } catch (const std::exception &) { failed = true; }
                            require(failed, "fault not reported");
                            require(llama_memory_seq_pos_max(mem, 0) == -1, "failed sequence not empty");
                            auto still_preserved = snapshot(mem, 1);
                            require(still_preserved.bytes == preserved.bytes, "unrelated sequence changed");
                            for (const auto & c : io.written) {
                                std::vector<uint8_t> data(c.size);
                                ggml_backend_tensor_get(c.tensor, data.data(), c.offset, c.size);
                                if (std::any_of(data.begin(), data.end(), [](uint8_t b) { return b != 0; })) {
                                    fprintf(stderr,"STALE %s gpu=%d unified=%d kv=%s tensor=%s offset=%zu size=%zu\n",fixture,gpu,unified,ggml_type_name(kv),c.tensor->name,c.offset,c.size);
                                    throw std::runtime_error("failed restore tensor data not zeroed");
                                }
                            }
                            auto result = decode(ctx.get(), 0, 0, 8);
                            for (float value : result) require(std::isfinite(value), "non-finite logits after failure");
                            llama_memory_clear(mem, true);
                            auto reference = decode(ctx.get(), 0, 0, 8);
                            double diff = 0, norm = 0;
                            for (size_t i = 0; i < result.size(); ++i) { diff += double(result[i]-reference[i])*(result[i]-reference[i]); norm += double(reference[i])*reference[i]; }
                            require(diff <= 1e-5*std::max(norm,1e-20), "new request differs from clean reference");
                            printf("PASS %s gpu=%d unified=%d kv=%s fault=%zu/%zu written=%zu NMSE=%g\n",fixture,gpu,unified,ggml_type_name(kv),fault,source.tensors.size(),io.written.size(),diff/std::max(norm,1e-20));
                            fflush(stdout); ++passed;
                        }
                    }
                }
            }
        }
        printf("SUMMARY passed=%d unsupported=%d\n",passed,skipped);
        return 0;
    } catch (const std::exception & error) { fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
