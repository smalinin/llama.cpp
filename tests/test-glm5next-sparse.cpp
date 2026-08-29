#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
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

static bool run_indexed_test(ggml_backend_t backend, int64_t n_query, ggml_type kv_type) {
    constexpr int64_t d         = 512;
    constexpr int64_t n_head    = 4;
    constexpr int64_t n_kv      = 512;
    constexpr int64_t n_stream  = 2;
    constexpr int64_t n_compact = 256;
    constexpr int64_t n_valid   = n_compact;

    ggml_init_params params = {
        /*.mem_size   =*/ 2u*1024u*1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_query, n_head, n_stream);
    ggml_tensor * kv = ggml_new_tensor_4d(ctx, kv_type, d, n_kv, 1, n_stream);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_query, 1, n_stream);
    ggml_tensor * idx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_compact, n_query, n_stream);
    ggml_tensor * compact_valid = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, n_compact, n_query, 1, n_stream);
    ggml_set_input(q);
    ggml_set_input(kv);
    ggml_set_input(mask);
    ggml_set_input(idx);
    ggml_set_input(compact_valid);

    ggml_tensor * out_dense = ggml_flash_attn_ext(ctx, q, kv, kv, mask,
            1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_dense, GGML_PREC_F32);

    ggml_tensor * out_indexed = ggml_flash_attn_ext(ctx, q, kv, kv, compact_valid,
            1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_indices(out_indexed, idx);
    ggml_flash_attn_ext_set_prec(out_indexed, GGML_PREC_F32);

    ggml_set_output(out_dense);
    ggml_set_output(out_indexed);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_dense);
    ggml_build_forward_expand(gf, out_indexed);

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

    std::vector<float> kv_f32(ggml_nelements(kv));
    for (size_t i = 0; i < kv_f32.size(); ++i) {
        kv_f32[i] = std::cos((float) i*0.007f);
    }
    if (kv_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> kv_data(kv_f32.size());
        for (size_t i = 0; i < kv_data.size(); ++i) {
            kv_data[i] = ggml_fp32_to_fp16(kv_f32[i]);
        }
        ggml_backend_tensor_set(kv, kv_data.data(), 0, kv_data.size()*sizeof(ggml_fp16_t));
    } else {
        GGML_ASSERT(kv_type == GGML_TYPE_Q8_0);
        std::vector<uint8_t> kv_data(ggml_nbytes(kv));
        const int64_t n_rows = ggml_nelements(kv)/kv->ne[0];
        GGML_ASSERT(ggml_quantize_chunk(kv_type, kv_f32.data(), kv_data.data(), 0, n_rows, kv->ne[0], nullptr) == kv_data.size());
        ggml_backend_tensor_set(kv, kv_data.data(), 0, kv_data.size());
    }

    std::vector<ggml_fp16_t> mask_data(ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
    std::vector<int32_t> idx_data(ggml_nelements(idx), 0);
    std::vector<ggml_fp16_t> valid_data(
            ggml_nelements(compact_valid), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t iq = 0; iq < n_query; ++iq) {
            const int64_t qoff = (s*n_query + iq)*n_compact;
            const int64_t moff = (s*n_query + iq)*n_kv;
            for (int64_t i = 0; i < n_valid; ++i) {
                const int32_t cell = (int32_t) ((i*7 + iq*13 + s*19)%n_kv);
                idx_data[qoff + i] = cell;
                valid_data[qoff + i] = ggml_fp32_to_fp16(0.0f);
                mask_data[moff + cell] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size()*sizeof(int32_t));
    ggml_backend_tensor_set(compact_valid, valid_data.data(), 0,
            valid_data.size()*sizeof(ggml_fp16_t));

    const bool computed = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    bool passed = computed;
    if (computed) {
        std::vector<float> dense(ggml_nelements(out_dense));
        std::vector<float> indexed(ggml_nelements(out_indexed));
        ggml_backend_tensor_get(out_dense, dense.data(), 0, dense.size()*sizeof(float));
        ggml_backend_tensor_get(out_indexed, indexed.data(), 0, indexed.size()*sizeof(float));

        float max_abs = 0.0f;
        for (size_t i = 0; i < dense.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs(dense[i] - indexed[i]));
        }
        passed = max_abs < 2e-3f;
        printf("%s: dense/indexed %s q=%" PRId64 " max abs = %.6g\n",
                ggml_backend_name(backend), ggml_type_name(kv_type), n_query, (double) max_abs);
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
            passed &= run_indexed_test(gpu, 4, GGML_TYPE_F16);
            passed &= run_indexed_test(gpu, 8, GGML_TYPE_F16);
            passed &= run_indexed_test(gpu, 4, GGML_TYPE_Q8_0);
            passed &= run_indexed_test(gpu, 8, GGML_TYPE_Q8_0);
            ggml_backend_free(gpu);
        }
    }

    llama_backend_free();
    return passed ? 0 : 1;
}
