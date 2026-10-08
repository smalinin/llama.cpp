static void scalar_routed_down(ggml_tensor * tensor, PrecisionOptions & options, const DownInput & input) {
    auto * view=tensor->src[1];
    check(view && view->op==GGML_OP_VIEW, "missing last expert view");
    auto * weighted=view->src[0];
    check(weighted && weighted->op==GGML_OP_MUL, "missing expert weights");
    auto * down=weighted->src[0];
    check(down && down->op==GGML_OP_MUL_MAT_ID, "missing routed down");
    auto * hidden=down->src[1];
    auto * ids=down->src[2];
    auto * weights=weighted->src[1];
    const int width=tensor->ne[1];
    check(width>1 && width<=4 && hidden->ne[2]==width && hidden->ne[1]==6 &&
            weights->ne[0]==1 && weights->ne[1]==6 && weights->ne[2]==width && ggml_is_contiguous(tensor), "unsupported down shape");
    auto device=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend=ggml_backend_dev_init(device,nullptr);
    check(backend,"scalar down backend failed");
    auto ctx=ggml_init({2*1024*1024,nullptr,true});
    check(ctx,"scalar down context failed");
    auto borrow=[&](ggml_tensor * src) {
        auto * leaf=ggml_dup_tensor(ctx,src);
        std::memcpy(leaf->nb,src->nb,sizeof(leaf->nb));leaf->data=src->data;leaf->buffer=src->buffer;
        return leaf;
    };
    auto * h=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hidden->ne[0],6,width);
    auto * all_ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,width);
    auto * all_weights=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,6,width);
    auto * w=borrow(down->src[0]);
    auto * graph=ggml_new_graph(ctx);
    ggml_tensor * result=nullptr;
    for (int col=0;col<width;++col) {
        auto * hc=ggml_view_3d(ctx,h,h->ne[0],h->ne[1],1,h->nb[1],h->nb[2],col*h->nb[2]);
        auto * ic=ggml_view_2d(ctx,all_ids,6,1,all_ids->nb[1],col*all_ids->nb[1]);
        auto * wc=ggml_view_3d(ctx,all_weights,1,6,1,all_weights->nb[1],all_weights->nb[2],col*all_weights->nb[2]);
        auto * dc=ggml_mul_mat_id(ctx,w,hc,ic);
        auto * ew=ggml_mul(ctx,dc,wc);
        ggml_build_forward_expand(graph,ew);
        std::vector<ggml_tensor *> views;
        for (int i=0;i<6;++i) {
            auto * v=ggml_view_2d(ctx,ew,dc->ne[0],1,ew->nb[2],i*ew->nb[1]);
            ggml_build_forward_expand(graph,v);views.push_back(v);
        }
        auto * y=views[0];
        for (int i=1;i<6;++i) {y=ggml_add(ctx,y,views[i]);ggml_build_forward_expand(graph,y);}
        result=result?ggml_concat(ctx,result,y,1):y;
    }
    ggml_build_forward_expand(graph,result);
    check(ggml_graph_n_nodes(graph)<=20*width,"scalar down escaped source isolation");
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    check(buffer,"scalar down allocation failed");
    check(input.hidden.size()==size_t(hidden->ne[0]*6*width) && input.ids.size()==size_t(6*width) && input.weights.size()==size_t(6*width),"cached down inputs incomplete");
    ggml_backend_tensor_set(h,input.hidden.data(),0,input.hidden.size()*4);
    ggml_backend_tensor_set(all_ids,input.ids.data(),0,input.ids.size()*4);
    ggml_backend_tensor_set(all_weights,input.weights.data(),0,input.weights.size()*4);
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"scalar down compute failed");
    check(ggml_nbytes(result)==ggml_nbytes(tensor),"scalar down result shape differs");
    std::vector<float> values(ggml_nelements(result));
    ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
    ggml_backend_tensor_set(tensor,values.data(),0,values.size()*4);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    options.matmul_counts["scalar-down-"+std::string(tensor->name)]+=width;
}
