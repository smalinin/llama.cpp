#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool run_test(ggml_backend_t backend) {
    constexpr int64_t d         = 512;
    constexpr int64_t n_head    = 4;
    constexpr int64_t n_kv      = 512;
    constexpr int64_t n_stream  = 2;
    constexpr int64_t n_compact = 256;

    ggml_init_params params = {
        /*.mem_size   =*/ 2u*1024u*1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, n_stream);
    ggml_tensor * kv   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 1, n_kv, n_stream);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 1, 1, n_stream);
    ggml_tensor * idx  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_compact, n_stream);
    ggml_tensor * compact_valid = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, n_compact, 1, 1, n_stream);
    ggml_set_input(q);
    ggml_set_input(kv);
    ggml_set_input(mask);
    ggml_set_input(idx);
    ggml_set_input(compact_valid);

    ggml_tensor * q_dense = ggml_permute(ctx,
            ggml_reshape_4d(ctx, q, d, n_head, 1, n_stream), 0, 2, 1, 3);
    ggml_tensor * kv_dense = ggml_permute(ctx, kv, 0, 2, 1, 3);
    ggml_tensor * out_dense = ggml_flash_attn_ext(ctx, q_dense, kv_dense, kv_dense, mask,
            1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_dense, GGML_PREC_F32);

    ggml_tensor * kv_rows = ggml_view_3d(ctx, kv, d, n_kv, n_stream, kv->nb[2], kv->nb[3], 0);
    ggml_tensor * kv_compact = ggml_get_rows(ctx, kv_rows, idx);
    kv_compact = ggml_reshape_4d(ctx, kv_compact, d, 1, n_compact, n_stream);
    kv_compact = ggml_permute(ctx, kv_compact, 0, 2, 1, 3);

    ggml_tensor * mask_cont = ggml_cont(ctx, mask);
    ggml_tensor * mask_rows = ggml_view_3d(ctx, mask_cont, 1, n_kv, n_stream,
            mask_cont->nb[0], mask_cont->nb[1], 0);
    ggml_tensor * mask_compact = ggml_get_rows(ctx, mask_rows, idx);
    mask_compact = ggml_reshape_4d(ctx, mask_compact, n_compact, 1, 1, n_stream);
    mask_compact = ggml_cast(ctx, mask_compact, GGML_TYPE_F16);
    mask_compact = ggml_add(ctx, mask_compact, compact_valid);

    ggml_tensor * q_compact = ggml_permute(ctx,
            ggml_reshape_4d(ctx, q, d, n_head, 1, n_stream), 0, 2, 1, 3);
    ggml_tensor * out_compact = ggml_flash_attn_ext(ctx, q_compact, kv_compact, kv_compact, mask_compact,
            1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_compact, GGML_PREC_F32);

    ggml_set_output(out_dense);
    ggml_set_output(out_compact);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_dense);
    ggml_build_forward_expand(gf, out_compact);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data(ggml_nelements(q));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = std::sin((float) i*0.013f);
    }
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(float));

    std::vector<ggml_fp16_t> kv_data(ggml_nelements(kv));
    for (size_t i = 0; i < kv_data.size(); ++i) {
        kv_data[i] = ggml_fp32_to_fp16(std::cos((float) i*0.007f));
    }
    ggml_backend_tensor_set(kv, kv_data.data(), 0, kv_data.size()*sizeof(ggml_fp16_t));

    const int valid[][6] = {
        { 0, 3, 5, 7, 11, 19 },
        { 1, 2, 8, 13, 21, 30 },
    };
    std::vector<ggml_fp16_t> mask_data(ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int j : valid[s]) {
            mask_data[s*n_kv + j] = ggml_fp32_to_fp16(0.0f);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(ggml_fp16_t));

    std::vector<int32_t> idx_data(ggml_nelements(idx), 0);
    std::vector<ggml_fp16_t> valid_data(
            ggml_nelements(compact_valid), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t i = 0; i < 6; ++i) {
            idx_data[s*n_compact + i] = valid[s][i];
            valid_data[s*n_compact + i] = ggml_fp32_to_fp16(0.0f);
        }
    }
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size()*sizeof(int32_t));
    ggml_backend_tensor_set(compact_valid, valid_data.data(), 0,
            valid_data.size()*sizeof(ggml_fp16_t));

    const bool computed = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    bool passed = computed;
    if (computed) {
        std::vector<float> dense(ggml_nelements(out_dense));
        std::vector<float> compact(ggml_nelements(out_compact));
        ggml_backend_tensor_get(out_dense, dense.data(), 0, dense.size()*sizeof(float));
        ggml_backend_tensor_get(out_compact, compact.data(), 0, compact.size()*sizeof(float));

        float max_abs = 0.0f;
        for (size_t i = 0; i < dense.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs(dense[i] - compact[i]));
        }
        passed = max_abs < 2e-3f;
        printf("%s: dense/compact max abs = %.6g\n", ggml_backend_name(backend), (double) max_abs);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int main() {
    llama_backend_init();

    bool passed = true;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    passed &= run_test(cpu);
    ggml_backend_free(cpu);

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev != nullptr) {
        ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
        if (gpu != nullptr) {
            passed &= run_test(gpu);
            ggml_backend_free(gpu);
        }
    }

    llama_backend_free();
    return passed ? 0 : 1;
}
