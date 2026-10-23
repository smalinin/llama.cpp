#include "models.h"

ggml_tensor * clip_graph_minimax_m3::apply_rope(
        ggml_tensor * x, ggml_tensor * pos_t, ggml_tensor * pos_h, ggml_tensor * pos_w) {
    const int dh  = (int) x->ne[0];
    const int axd = 2 * ((2 * (dh / 2) / 3) / 2);

    GGML_ASSERT(3 * axd <= dh);

    const float th  = hparams.rope_theta;

    x = ggml_rope_ext(ctx0, x, pos_t, nullptr, axd, GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    x = ggml_rope_ext(ctx0, x, pos_h, nullptr, axd, GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    x = ggml_rope_set_offset(x, axd);

    x = ggml_rope_ext(ctx0, x, pos_w, nullptr, axd, GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    x = ggml_rope_set_offset(x, 2 * axd);

    return x;
}

ggml_cgraph * clip_graph_minimax_m3::build() {
    GGML_ASSERT(model.patch_bias     == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.patch_embeddings_0 && model.patch_embeddings_1);
    GGML_ASSERT(model.mm_1_w && model.mm_2_w);
    GGML_ASSERT(model.mm_merger_fc1_w && model.mm_merger_fc2_w);

    GGML_ASSERT(n_batch >= 1 && n_batch <= 8);
    const int n_temporal = (n_batch + 1) / 2;
    const int n_pos      = n_patches * n_temporal;
    const int merge      = hparams.n_merge;

    // patch embedding
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = nullptr;
    for (int t = 0; t < n_temporal; ++t) {
        const int frame0 = 2 * t;
        const int frame1 = std::min(frame0 + 1, n_batch - 1);
        ggml_tensor * raw0 = ggml_view_3d(ctx0, inp_raw, img.nx(), img.ny(), 3, inp_raw->nb[1], inp_raw->nb[2], frame0 * inp_raw->nb[3]);
        ggml_tensor * raw1 = ggml_view_3d(ctx0, inp_raw, img.nx(), img.ny(), 3, inp_raw->nb[1], inp_raw->nb[2], frame1 * inp_raw->nb[3]);
        ggml_tensor * patches = ggml_add(ctx0,
            ggml_conv_2d(ctx0, model.patch_embeddings_0, raw0, patch_size, patch_size, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, model.patch_embeddings_1, raw1, patch_size, patch_size, 0, 0, 1, 1));

        patches = ggml_permute(ctx0, patches, 1, 2, 0, 3);
        patches = ggml_cont_4d(ctx0, patches, n_embd * merge, n_patches_x / merge, n_patches_y, 1);
        patches = ggml_reshape_4d(ctx0, patches, n_embd * merge, n_patches_x / merge, merge, n_patches_y / merge);
        patches = ggml_permute(ctx0, patches, 0, 2, 1, 3);
        patches = ggml_cont_3d(ctx0, patches, n_embd, n_patches, 1);
        inp = inp ? ggml_concat(ctx0, inp, patches, 1) : patches;
    }

    ggml_tensor * pos_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_t, "minimax_pos_t"); ggml_set_input(pos_t);
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "minimax_pos_h"); ggml_set_input(pos_h);
    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "minimax_pos_w"); ggml_set_input(pos_w);

    ggml_tensor * inpL = build_vit(
        inp, n_pos, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr,
        [&](ggml_tensor * c, const clip_layer &) {
            return apply_rope(c, pos_t, pos_h, pos_w);
        });

    // projector
    ggml_tensor * emb = inpL;
    emb = build_ffn(emb, model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b, FFN_GELU_ERF, -1);

    const int64_t proj = emb->ne[0];
    emb = ggml_reshape_2d(ctx0, emb, proj * merge * merge, n_pos / (merge * merge));

    emb = build_ffn(emb, model.mm_merger_fc1_w, model.mm_merger_fc1_b,
                    nullptr, nullptr,
                    model.mm_merger_fc2_w, model.mm_merger_fc2_b, FFN_GELU_ERF, -1);

    ggml_build_forward_expand(gf, emb);
    return gf;
}
