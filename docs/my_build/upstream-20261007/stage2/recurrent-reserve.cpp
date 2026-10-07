#include "llama.h"
#include "llama-graph.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * msg) { if (!ok) { throw std::runtime_error(msg); } }
struct graph_case {
    llm_graph_result result{128};
    ggml_tensor * ids;
    ggml_tensor * output;
    graph_case(const llama_hparams & hparams, ggml_tensor * cache, int n_seqs, bool custom) {
        llm_graph_params params{};
        params.hparams = hparams; params.res = &result; params.ubatch.n_tokens = n_seqs;
        params.ubatch.n_seqs = n_seqs;
        llm_graph_context graph(params);
        ids = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, 4);
        ggml_set_input(ids);
        auto * main = ggml_view_1d(graph.ctx0, ids, n_seqs, 0);
        llm_graph_get_rows_fn getter = custom ? llm_graph_get_rows_fn(ggml_get_rows) : llm_graph_get_rows_fn{};
#ifdef BEFORE_STAGE2
        auto * extra = ggml_view_1d(graph.ctx0, ids, 4 - n_seqs, n_seqs * ids->nb[0]);
        output = graph.build_rs(cache, main, extra, cache->ne[0], n_seqs, 4, 0, 8, -1,
                                custom ? getter : llm_graph_get_rows_fn(ggml_get_rows));
#else
        auto * copy = custom ? ggml_view_1d(graph.ctx0, ids, 3, ids->nb[0]) : ids;
        output = graph.build_rs(cache, copy, main, cache->ne[0], n_seqs, 4, 0, 8, -1, getter);
#endif
        ggml_set_output(output);
    }
};
int main(int argc, char ** argv) {
    try {
        (void) argc; (void) argv;
        llama_hparams hparams{};
        hparams.n_layer_all = 1;
        hparams.n_embd = 32;
        hparams.n_head_arr[0] = hparams.n_head_kv_arr[0] = 1;
        hparams.n_embd_head_k_full = hparams.n_embd_head_v_full = 32;
        ggml_backend_t cpu = ggml_backend_cpu_init();
        // Two backends prevent the allocator from silently reserving a single-buffer graph.
        ggml_backend_t cpu_second = ggml_backend_cpu_init();
        ggml_backend_t backends[] = {cpu, cpu_second};
        ggml_init_params init{ggml_tensor_overhead(), nullptr, true};
        ggml_context * cache_ctx = ggml_init(init);
        auto * cache = ggml_new_tensor_2d(cache_ctx, GGML_TYPE_F32, 4096, 8);
        auto * buffer = ggml_backend_alloc_ctx_tensors(cache_ctx, cpu);
        require(buffer, "cache allocation failed");
        const int ids[] = {5, 3, 6, 2};
        std::vector<float> data(4096 * 8);
        for (size_t i = 0; i < data.size(); ++i) { data[i] = float(i); }
        for (bool custom : {false, true}) {
            auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
            graph_case reserve(hparams, cache, 4, custom);
            require(ggml_backend_sched_reserve(sched, reserve.result.get_gf()), "reserve failed");
            require(ggml_backend_sched_alloc_graph(sched, reserve.result.get_gf()), "reserve graph allocation failed");
            ggml_backend_tensor_set(cache, data.data(), 0, data.size() * sizeof(float));
            ggml_backend_tensor_set(reserve.ids, ids, 0, sizeof(ids));
            require(ggml_backend_sched_graph_compute(sched, reserve.result.get_gf()) == GGML_STATUS_SUCCESS, "reserve graph failed");
            ggml_backend_sched_reset(sched);
            for (int n_seqs : {1, 2, 3, 4}) {
                graph_case actual(hparams, cache, n_seqs, custom);
                require(ggml_backend_sched_alloc_graph(sched, actual.result.get_gf()), "split graph allocation failed");
                ggml_backend_tensor_set(cache, data.data(), 0, data.size() * sizeof(float));
                ggml_backend_tensor_set(actual.ids, ids, 0, sizeof(ids));
                require(ggml_backend_sched_graph_compute(sched, actual.result.get_gf()) == GGML_STATUS_SUCCESS, "split graph failed");
                std::vector<float> output(4096 * n_seqs), final_state(data.size());
                ggml_backend_tensor_get(actual.output, output.data(), 0, output.size() * sizeof(float));
                ggml_backend_tensor_get(cache, final_state.data(), 0, final_state.size() * sizeof(float));
                for (int row = 0; row < n_seqs; ++row) {
                    for (int k = 0; k < 4096; ++k) { require(output[row * 4096 + k] == data[ids[row] * 4096 + k], "main state mismatch"); }
                }
                for (int row = n_seqs; row < 4; ++row) {
                    for (int k = 0; k < 4096; ++k) { require(final_state[row * 4096 + k] == data[ids[row] * 4096 + k], "extra state mismatch"); }
                }
                printf("PASS custom=%d n_seqs=%d n_rs=4\n", custom, n_seqs); fflush(stdout);
                ggml_backend_sched_reset(sched);
            }
            ggml_backend_sched_free(sched);
        }
        ggml_backend_buffer_free(buffer); ggml_free(cache_ctx); ggml_backend_free(cpu); ggml_backend_free(cpu_second);
        puts("RESULT passed=8");
    } catch (const std::exception & err) { fprintf(stderr, "FAIL %s\n", err.what()); return 1; }
}
