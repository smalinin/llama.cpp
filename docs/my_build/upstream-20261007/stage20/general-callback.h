#pragma once
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ds14 {
static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void logging(ggml_log_level level, const char * text, void *) {
    if (level <= GGML_LOG_LEVEL_WARN || std::string(text).find("load_tensors:") != std::string::npos) {
        std::fputs(text, stderr);
    }
}

static std::string read_text(const char * path) {
    std::ifstream input(path);
    check(bool(input), "cannot read input");
    return {std::istreambuf_iterator<char>(input), {}};
}


struct PrecisionOptions {
    std::string mode;
    bool features;
    bool decoding = false;
    int replaced = 0;
    int replaced_upgate = 0;
    int precision_requests = 0;
    int absolute_position = -1;
    int raw_capacity = 0;
    int raw_limit = 0;
    std::vector<int> compress_ratios;
    int padding_crops = 0;
    int compressed_padding_crops = 0;
    std::string directory, label;
    std::vector<std::string> captured;
    std::map<std::string,int> matmul_counts;
};

static void scalar_attention(ggml_tensor * tensor, PrecisionOptions & options) {
    auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend, "scalar attention backend failed");
    auto * source_q = tensor->src[0];
    auto * source_mask = tensor->src[3];
    check(source_mask && source_mask->ne[2] == 1 && source_q->ne[3] == 1, "unsupported attention shape");
    const int layer = std::stoi(std::string(source_q->name).substr(2));
    check(layer >= 0 && layer < (int) options.compress_ratios.size(),"missing compression ratio");
    if (layer == 0) {
        options.raw_capacity = tensor->src[1]->ne[1];
        options.raw_limit = tensor->src[1]->nb[3]/tensor->src[1]->nb[1];
        check(options.raw_limit >= options.raw_capacity && options.raw_limit%256 == 0,"unsupported raw storage");
    }
    for (int col = 0; col < source_q->ne[1]; ++col) {
        auto ctx = ggml_init({2*1024*1024, nullptr, true});
        check(ctx, "scalar attention context failed");
        auto borrow = [&](ggml_tensor * src) {
            auto * leaf = ggml_dup_tensor(ctx, src);
            std::memcpy(leaf->nb, src->nb, sizeof(leaf->nb));
            leaf->data = src->data;
            leaf->buffer = src->buffer;
            return leaf;
        };
        auto * q_leaf = borrow(source_q);
        auto * mask_leaf = borrow(source_mask);
        auto * q = ggml_view_4d(ctx, q_leaf, source_q->ne[0], 1, source_q->ne[2], 1,
                source_q->nb[1], source_q->nb[2], source_q->nb[3], col*source_q->nb[1]);
        auto * mask = ggml_view_4d(ctx, mask_leaf, source_mask->ne[0], 1, 1, 1,
                source_mask->nb[1], source_mask->nb[2], source_mask->nb[3], col*source_mask->nb[1]);
        float scale, max_bias, cap;
        std::memcpy(&scale, tensor->op_params, 4);
        std::memcpy(&max_bias, tensor->op_params+1, 4);
        std::memcpy(&cap, tensor->op_params+2, 4);
        auto * k = borrow(tensor->src[1]);
        auto * v = borrow(tensor->src[2]);
        // This probe assumes a fresh, append-only sequence with no compaction.
        check(options.absolute_position >= 0,"missing absolute query position");
        const int pos = options.absolute_position+col;
        const int raw = options.raw_capacity;
        const int compressed = k->ne[1]-raw;
        const int ratio = options.compress_ratios[layer];
        const auto pad256 = [](int n) { return std::max(256,((n+255)/256)*256); };
        const int keep_raw = std::min(options.raw_limit,pad256(pos+1));
        const int keep_comp = ratio ? pad256((pos+1)/ratio) : 0;
        check(k->ne[2] == 1 && v->ne[2] == 1 && raw >= keep_raw && compressed >= keep_comp,"unsupported cache extent");
        check(mask->ne[0] == raw+compressed,"mask extent mismatch");
        if (raw != keep_raw || compressed != keep_comp) {
            auto check_tail = [&](int start, int count) {
                std::vector<ggml_fp16_t> tail(count);
                if (count) ggml_backend_tensor_get(source_mask,tail.data(),col*source_mask->nb[1]+start*2,count*2);
                for (auto x : tail) check(std::isinf(ggml_fp16_to_fp32(x)) && ggml_fp16_to_fp32(x) < 0,"crop would remove a visible row");
            };
            check_tail(keep_raw,raw-keep_raw);
            check_tail(raw+keep_comp,compressed-keep_comp);
            auto crop_kv = [&](ggml_tensor * leaf) {
                auto * first = ggml_view_4d(ctx,leaf,leaf->ne[0],keep_raw,1,1,leaf->nb[1],leaf->nb[2],leaf->nb[3],0);
                if (!keep_comp) return first;
                auto * rest = ggml_view_4d(ctx,leaf,leaf->ne[0],keep_comp,1,1,leaf->nb[1],leaf->nb[2],leaf->nb[3],raw*leaf->nb[1]);
                return ggml_concat(ctx,first,rest,1);
            };
            k = crop_kv(k);
            v = tensor->src[1] == tensor->src[2] ? k : crop_kv(v);
            auto * first = ggml_view_4d(ctx,mask,keep_raw,1,1,1,mask->nb[1],mask->nb[2],mask->nb[3],0);
            if (!keep_comp) mask = first;
            else {
                auto * rest = ggml_view_4d(ctx,mask,keep_comp,1,1,1,mask->nb[1],mask->nb[2],mask->nb[3],raw*mask->nb[0]);
                mask = ggml_concat(ctx,first,rest,0);
            }
            if (raw != keep_raw) ++options.padding_crops;
            if (compressed != keep_comp) ++options.compressed_padding_crops;
        }
        auto * y = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, cap);
        if (tensor->src[4]) ggml_flash_attn_ext_add_sinks(y, borrow(tensor->src[4]));
        ggml_flash_attn_ext_set_prec(y, ggml_flash_attn_ext_get_prec(tensor));
        ggml_flash_attn_ext_set_n_kv_max(y, tensor->op_params[4]);
        check(tensor->src[5] == nullptr, "indexed attention not supported by this probe");
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        check(ggml_graph_n_nodes(graph) <= 16, "scalar graph escaped source isolation");
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer, "scalar attention allocation failed");
        check(ggml_backend_graph_compute(backend, graph)==GGML_STATUS_SUCCESS, "scalar attention failed");
        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, out.size()*4);
        ggml_backend_tensor_set(tensor, out.data(), col*out.size()*4, out.size()*4);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ++options.replaced;
    }
    ggml_backend_free(backend);
}

static void scalar_upgate(ggml_tensor * tensor, PrecisionOptions & options) {
    auto * gate_op = tensor->src[0];
    auto * up_op = tensor->src[1];
    check(tensor->op == GGML_OP_GLU && ggml_get_glu_op(tensor) == GGML_GLU_OP_SWIGLU_CLAMP,
          "unsupported routed GLU");
    check(gate_op->op == GGML_OP_MUL_MAT_ID && up_op->op == GGML_OP_MUL_MAT_ID,
          "unsupported routed projections");
    check(gate_op->src[1] == up_op->src[1] && gate_op->src[2] == up_op->src[2],
          "routed projections have different inputs");
    auto * source_x = gate_op->src[1];
    auto * source_ids = gate_op->src[2];
    const int width = source_x->ne[2];
    check(width > 1 && width <= 4 && ggml_is_contiguous(tensor), "unsupported routed batch");
    auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend, "scalar upgate backend failed");
    auto ctx = ggml_init({2*1024*1024, nullptr, true});
    check(ctx, "scalar upgate context failed");
    auto borrow = [&](ggml_tensor * src) {
        auto * leaf = ggml_dup_tensor(ctx, src);
        std::memcpy(leaf->nb, src->nb, sizeof(leaf->nb));
        leaf->data = src->data;
        leaf->buffer = src->buffer;
        return leaf;
    };
    auto * x = borrow(source_x);
    auto * ids = borrow(source_ids);
    auto * gate_weight = borrow(gate_op->src[0]);
    auto * up_weight = borrow(up_op->src[0]);
    float limit;
    std::memcpy(&limit, tensor->op_params+3, 4);
    ggml_tensor * result = nullptr;
    for (int col = 0; col < width; ++col) {
        auto * x_col = ggml_view_3d(ctx, x, x->ne[0], 1, 1, x->nb[1], x->nb[2], col*x->nb[2]);
        auto * ids_col = ggml_view_2d(ctx, ids, ids->ne[0], 1, ids->nb[1], col*ids->nb[1]);
        auto * up = ggml_mul_mat_id(ctx, up_weight, x_col, ids_col);
        auto * gate = ggml_mul_mat_id(ctx, gate_weight, x_col, ids_col);
        auto * y = ggml_swiglu_clamp(ctx, gate, up, limit);
        result = result ? ggml_concat(ctx, result, y, 2) : y;
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    check(ggml_graph_n_nodes(graph) <= 6*width, "scalar upgate escaped source isolation");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer, "scalar upgate allocation failed");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "scalar upgate failed");
    check(ggml_nbytes(result) == ggml_nbytes(tensor), "scalar upgate shape differs");
    std::vector<float> out(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, out.size()*4);
    ggml_backend_tensor_set(tensor, out.data(), 0, out.size()*4);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    options.replaced_upgate += width;
}

static void capture_first(ggml_tensor * tensor, PrecisionOptions & options, const std::string & phase) {
    if (options.directory.empty()) return;
    const std::string name = tensor->name;
    if (name != "ffn_moe_swiglu_limited-2") return;
    const std::string key = name+"-"+phase;
    if (std::find(options.captured.begin(), options.captured.end(), key) != options.captured.end()) return;
    options.captured.push_back(key);
    check(ggml_is_contiguous(tensor) && tensor->type == GGML_TYPE_F32, "unsupported capture");
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*4);
    std::ofstream(options.directory+"/"+options.label+"-"+key+".f32", std::ios::binary)
        .write(reinterpret_cast<const char *>(values.data()), values.size()*4);
}

static bool selected_float_matmul(ggml_tensor * tensor, const PrecisionOptions & options) {
    if (tensor->op != GGML_OP_MUL_MAT || tensor->ne[1] <= 1 || tensor->ne[1] > 4) return false;
    auto * w = tensor->src[0];
    auto * x = tensor->src[1];
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_BF16) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const std::string name = tensor->name;
    return options.mode.find("float2d") != std::string::npos ||
        (options.mode.find("compressor") != std::string::npos &&
         (name.find("comp_state_kv-") == 0 || name.find("comp_state_score-") == 0)) ||
        (options.mode.find("router") != std::string::npos && name.find("ffn_moe_logits-") == 0) ||
        (options.mode.find("hc-router") != std::string::npos && name.find("hc_mixes-") == 0);
}

static void scalar_float_matmul(ggml_tensor * tensor, PrecisionOptions & options) {
    auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    auto backend = ggml_backend_dev_init(device, nullptr);
    check(backend, "scalar float backend failed");
    auto ctx = ggml_init({2*1024*1024, nullptr, true});
    auto borrow = [&](ggml_tensor * src) {
        auto * leaf = ggml_dup_tensor(ctx, src);
        std::memcpy(leaf->nb, src->nb, sizeof(leaf->nb));
        leaf->data = src->data;
        leaf->buffer = src->buffer;
        return leaf;
    };
    auto * w = borrow(tensor->src[0]);
    auto * x = borrow(tensor->src[1]);
    ggml_tensor * result = nullptr;
    const int width = tensor->ne[1];
    for (int col = 0; col < width; ++col) {
        auto * x_col = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], col*x->nb[1]);
        auto * y = ggml_mul_mat(ctx, w, x_col);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        result = result ? ggml_concat(ctx, result, y, 1) : y;
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    check(ggml_graph_n_nodes(graph) <= 3*width, "scalar float escaped source isolation");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer, "scalar float allocation failed");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "scalar float failed");
    check(ggml_nbytes(result) == ggml_nbytes(tensor) && ggml_is_contiguous(tensor), "scalar float shape differs");
    std::vector<float> out(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, out.size()*4);
    ggml_backend_tensor_set(tensor, out.data(), 0, out.size()*4);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    options.matmul_counts[tensor->name] += width;
}

static bool controlled_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & options = *static_cast<PrecisionOptions *>(data);
    if (!options.decoding) return false;
    if (ask && tensor->op == GGML_OP_MUL_MAT && options.mode.find("decode") == 0) {
        const auto type = tensor->src[0]->type;
        if (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) { ggml_mul_mat_set_prec(tensor,GGML_PREC_F32); ++options.precision_requests; }
    }
    const bool attention = tensor->op == GGML_OP_FLASH_ATTN_EXT;
    const bool upgate = tensor->op == GGML_OP_GLU && std::string(tensor->name).find("ffn_moe_swiglu_limited-") == 0;
    const bool matmul = selected_float_matmul(tensor, options);
    if (ask) return attention || upgate || matmul;
    if (matmul) scalar_float_matmul(tensor, options);
    if (attention && options.mode.find("scalar-fa") != std::string::npos && tensor->src[0]->ne[1] > 1) scalar_attention(tensor, options);
    if (upgate) {
        capture_first(tensor, options, "before");
        if (options.mode.find("upgate") != std::string::npos && tensor->ne[2] > 1) scalar_upgate(tensor, options);
        capture_first(tensor, options, "after");
    }
    return true;
}

}

namespace ds14 {
struct DownInput {
    std::vector<float> hidden,weights;
    std::vector<int32_t> ids;
};
struct DownState {
    PrecisionOptions * precision;
    std::map<int,DownInput> inputs;
    std::string capture_prefix;
};
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

static bool down_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & state=*static_cast<DownState *>(data);
    auto & options=*state.precision;
    const bool requested=controlled_precision(tensor,ask,&options);
    if (!options.decoding) return requested;
    const std::string name=tensor->name;
    const bool weight=name.find("ffn_moe_weights_scaled-")==0;
    const bool hidden=name.find("ffn_moe_swiglu_limited-")==0;
    const bool output=name.find("ffn_moe_out-")==0;
    const bool capture=!state.capture_prefix.empty() && (name=="ffn_moe_out-0" || name=="ffn_out-0");
    if (ask) return requested || weight || output || capture;
    if (weight || hidden || output) {
        const int layer=std::stoi(name.substr(name.rfind('-')+1));
        auto & input=state.inputs[layer];
        if (weight && tensor->ne[2]>1) {
            check(ggml_is_contiguous(tensor),"non-contiguous expert weights");
            input.weights.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor,input.weights.data(),0,input.weights.size()*4);
        }
        if (hidden && tensor->ne[2]>1) {
            check(ggml_is_contiguous(tensor),"non-contiguous hidden");
            input.hidden.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor,input.hidden.data(),0,input.hidden.size()*4);
            auto * ids=tensor->src[0]->src[2];
            check(ids && ids->type==GGML_TYPE_I32 && ids->ne[0]==6,"unsupported cached IDs");
            input.ids.resize(6*tensor->ne[2]);
            for (int col=0;col<tensor->ne[2];++col) ggml_backend_tensor_get(ids,input.ids.data()+6*col,col*ids->nb[1],6*4);
        }
        if (output && tensor->ne[1]>1) scalar_routed_down(tensor,options,input);
    }
    if (capture) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);
        std::ofstream(state.capture_prefix+"-"+name+".f32",std::ios::binary).write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    }
    return true;
}

}
