#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-batch.h"
#include <cstdio>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static llama_kv_cache::slot_info place(llama_kv_cache & cache, int pos, int count, int seq, bool apply) {
    llama_batch_allocr alloc(1);
    auto batch = alloc.ubatch_reserve(count, 1);
    batch.seq_id_unq[0] = seq;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = 0;
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = batch.seq_id_unq;
    }
    auto slot = cache.find_slot(batch, false);
    require(!slot.empty() && slot.idxs[0].size() == size_t(count), "slot allocation failed");
    if (apply) cache.apply_ubatch(slot, batch);
    return slot;
}
static bool run_case(const char * label, llm_arch arch, int start, int kept, int seq,
                     bool unified, int nseq, int remove_seq, int p0, int p1, int expected) {
    std::unique_ptr<llama_model> owner(llama_model_create(arch, llama_model_default_params()));
    auto & model = *owner;
    llama_kv_cache cache(model, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                         false, false, unified, 768, nseq, 1, 128, LLAMA_SWA_TYPE_STANDARD,
                         nullptr, nullptr, nullptr, nullptr);
    if (seq == 1) place(cache, 0, 10, 0, true);
    place(cache, 0, start, seq, true);
    const auto before = place(cache, start, 4, seq, true);
    require(cache.seq_rm(remove_seq, p0, p1), "remove failed");
    if (seq == 1) require(cache.seq_pos_max(0) == 9, "other sequence changed");
    const auto after = place(cache, p0, 4, seq, false);
    bool pass = after.idxs[0][0] == uint32_t(expected);
    if (remove_seq == seq && p0 == start + kept && p1 == -1) {
        const auto & cells = cache.get_cells(seq);
        for (int col = 0; col < kept; ++col) {
            const auto index = before.idxs[0][col];
            pass = pass && !cells.is_empty(index) && cells.pos_get(index) == start + col;
        }
        for (int col = kept; col < 4; ++col) pass = pass && cells.is_empty(before.idxs[0][col]);
    }
    std::printf("{\"case\":\"%s\",\"start\":%d,\"kept\":%d,\"expected_first\":%d,\"actual_first\":%u,\"passed\":%s}\n",
                label, start, kept, expected, after.idxs[0][0], pass ? "true" : "false");
    return pass;
}
int main() {
    try {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        int failures = 0, cases = 0;
        auto check = [&](const char * label, llm_arch arch, int start, int kept, int seq,
                         bool unified, int nseq, int remove_seq, int p0, int p1, int expected) {
            ++cases;failures += !run_case(label, arch, start, kept, seq, unified, nseq, remove_seq, p0, p1, expected);
        };
        for (int start : {100, 764, 765, 766, 767}) {
            for (int kept = 0; kept < 4; ++kept) {
                check("tail", LLM_ARCH_DEEPSEEK41, start, kept, 0, false, 1, 0, start + kept, -1, (start + kept) % 768);
            }
        }
        check("separate-stream", LLM_ARCH_DEEPSEEK41, 765, 1, 1, false, 2, 1, 766, -1, 766);
        check("other-architecture", LLM_ARCH_LLAMA, 765, 1, 0, false, 1, 0, 766, -1, 0);
        check("shared-stream", LLM_ARCH_DEEPSEEK41, 765, 1, 0, true, 2, 0, 766, -1, 0);
        check("finite-range", LLM_ARCH_DEEPSEEK41, 765, 1, 0, false, 1, 0, 766, 769, 0);
        check("wildcard", LLM_ARCH_DEEPSEEK41, 765, 1, 0, false, 1, -1, 766, -1, 0);
        check("full-clear", LLM_ARCH_DEEPSEEK41, 765, 0, 0, false, 1, 0, 0, -1, 0);
        check("empty-tail", LLM_ARCH_DEEPSEEK41, 765, 4, 0, false, 1, 0, 769, -1, 1);
        std::printf("{\"cases\":%d,\"failures\":%d}\n", cases, failures);
        return failures ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());return 2;
    }
}
