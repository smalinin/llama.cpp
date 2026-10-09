static void build_attn_ordered(ggml_context * ctx, ggml_tensor * order, ggml_tensor * valid,
        ggml_tensor * & k, ggml_tensor * & mask) {
    if (!order) {
        return;
    }
    GGML_ASSERT(k->ne[1] == 1 && mask != nullptr);
    const int64_t n_stream = k->ne[3];
    const int64_t n_kv = k->ne[2];
    const int64_t n_order = order->ne[0];
    const int64_t n_query = mask->ne[1];

    auto * rows = ggml_view_3d(ctx, k, k->ne[0], n_kv, n_stream, k->nb[2], k->nb[3], 0);
    k = ggml_get_rows(ctx, rows, order);
    k = ggml_reshape_4d(ctx, k, k->ne[0], 1, n_order, n_stream);

    auto * mask_rows = ggml_cont(ctx, ggml_permute(ctx, mask, 1, 0, 2, 3));
    mask_rows = ggml_reshape_3d(ctx, mask_rows, n_query, n_kv, n_stream);
    mask_rows = ggml_get_rows(ctx, mask_rows, order);
    mask_rows = ggml_cont(ctx, ggml_permute(ctx, mask_rows, 1, 0, 2, 3));
    mask_rows = ggml_reshape_4d(ctx, mask_rows, n_order, n_query, 1, n_stream);
    mask_rows = ggml_add(ctx, mask_rows, valid);
    mask = mask->type == GGML_TYPE_F16 ? ggml_cast(ctx, mask_rows, GGML_TYPE_F16) : mask_rows;
}
