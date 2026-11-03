#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using json = nlohmann::ordered_json;
static void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
static ggml_status (*original_compute)(ggml_backend_t, ggml_cgraph *);
static bool fail_once = false;
static ggml_status compute_fault(ggml_backend_t backend, ggml_cgraph * graph) {
    if (fail_once) { fail_once = false; return GGML_STATUS_FAILED; }
    return original_compute(backend, graph);
}
static bool callback(ggml_tensor *, bool, void *) { return false; }

struct Result {
    std::vector<std::vector<float>> norm, base;
    int aliases = 0, failures = 0;
    std::vector<size_t> sizes;
};
static Result run(ggml_backend_t a, ggml_backend_t b, ggml_backend_t cpu,
                  const std::string & variant, int mode, int steps) {
    if (mode < 0) unsetenv("GGML_SCHED_COPY_RESHAPE");
    else setenv("GGML_SCHED_COPY_RESHAPE", mode ? "1" : "0", 1);
    const bool parallel = variant != "single_copy";
    ggml_backend_t backends[] = {a, b, cpu};
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(a),
        ggml_backend_get_default_buffer_type(b), ggml_backend_dev_host_buffer_type(ggml_backend_get_device(a))};
    auto * sched = ggml_backend_sched_new(backends, bufts, 3, 2048, parallel, true);
    require(sched, "scheduler");
    if (parallel) require(ggml_backend_sched_set_inputs_async(sched, true), "async inputs");
    if (variant == "callback") ggml_backend_sched_set_eval_callback(sched, callback, nullptr);
    Result result;
    std::vector<ggml_context *> contexts;
    for (int step = 0; step < steps; ++step) {
        constexpr int d = 64, hc = 4;
        const int widths[] = {1,4,6,7,508,511,512,513};
        const int n = variant == "eligible" || variant == "failure" ? widths[(step / 20) % 8] : 17;
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        require(ctx, "context"); contexts.push_back(ctx);
        auto * graph = ggml_new_graph_custom(ctx, 2048, false);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
        auto * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, hc, n);
        auto * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n);
        auto * pre = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n);
        for (auto * t : {x, residual, post, pre}) ggml_set_input(t);
        auto * produced = ggml_dsv4_hc_post(ctx, x, residual, post, nullptr);
        auto * root = variant == "other_root" ? ggml_scale(ctx, produced, 1.0f) : produced;
        if (variant == "root_output") ggml_set_output(root);
        if (variant == "root_input") ggml_set_input(root);
        auto * base = ggml_dsv4_hc_pre(ctx, root, pre);
        ggml_set_output(base);
        auto * view = ggml_reshape_2d(ctx, root, d*hc, n);
        if (variant == "nested") view = ggml_reshape_4d(ctx, view, d*hc, n, 1, 1);
        if (variant == "offset") view = ggml_view_2d(ctx, root, d*hc, n-1, root->nb[2], root->nb[2]);
        if (variant == "subrange") view = ggml_view_2d(ctx, root, d*hc, n-1, root->nb[2], 0);
        if (variant == "strided") view = ggml_view_2d(ctx, root, d*hc, n/2, root->nb[2]*2, 0);
        if (variant == "view_output") ggml_set_output(view);
        auto * norm = ggml_rms_norm(ctx, view, 1.0e-5f);
        ggml_set_output(norm);
        ggml_tensor * extra = nullptr;
        if (variant == "other_reader") {
            extra = ggml_scale(ctx, root, 2.0f);
            ggml_set_output(extra); ggml_build_forward_expand(graph, extra);
        }
        if (variant == "late_root") {
            ggml_build_forward_expand(graph, norm); ggml_build_forward_expand(graph, base);
        } else {
            ggml_build_forward_expand(graph, base);
            if (variant == "different_split") {
                extra = ggml_scale(ctx, base, 2.0f);
                ggml_set_output(extra); ggml_build_forward_expand(graph, extra);
            }
            ggml_build_forward_expand(graph, norm);
        }
        ggml_backend_sched_reset(sched);
        for (auto * t : {x, residual, post, pre}) ggml_backend_sched_set_tensor_backend(sched, t, cpu);
        ggml_backend_sched_set_tensor_backend(sched, produced, a);
        ggml_backend_sched_set_tensor_backend(sched, root, a);
        ggml_backend_sched_set_tensor_backend(sched, view, a);
        auto * dest = variant == "same_device" ? a : b;
        ggml_backend_sched_set_tensor_backend(sched, base, dest);
        auto * norm_dest = variant == "cpu_destination" ? cpu : dest;
        ggml_backend_sched_set_tensor_backend(sched, norm, norm_dest);
        if (extra) ggml_backend_sched_set_tensor_backend(sched, extra, variant == "different_split" ? cpu : a);
        require(ggml_backend_sched_alloc_graph(sched, graph), "allocate graph");
        auto * actual = norm->src[0];
        const bool shared = actual != view && actual->op == GGML_OP_RESHAPE && actual->view_src == base->src[0];
        const bool expected_share = mode == 1 && (variant == "eligible" || variant == "failure");
        if (shared != expected_share) {
            std::fprintf(stderr, "branch mismatch variant=%s mode=%d step=%d src=%s op=%s root=%s\n",
                variant.c_str(), mode, step, actual->name, ggml_op_name(actual->op), base->src[0]->name);
            require(false, "alias eligibility");
        }
        if (shared) {
            require(actual->data == base->src[0]->data && actual->buffer == base->src[0]->buffer, "shared storage");
            require(ggml_is_contiguous(actual) && actual->ne[0] == d*hc && actual->ne[1] == n, "shared layout");
            ++result.aliases;
        }
        ggml_backend_sched_prepare_inputs(sched);
        std::vector<float> zeros_x(d*n, 0), zeros_post(hc*n, 0), pre_data(hc*n, .25f), r(d*hc*n);
        for (size_t i = 0; i < r.size(); ++i) r[i] = float(step+1) + float(i % 17)/16;
        ggml_backend_tensor_set(x, zeros_x.data(), 0, zeros_x.size()*4);
        ggml_backend_tensor_set(post, zeros_post.data(), 0, zeros_post.size()*4);
        ggml_backend_tensor_set(pre, pre_data.data(), 0, pre_data.size()*4);
        ggml_backend_tensor_set(residual, r.data(), 0, r.size()*4);
        if (variant == "failure" && step == 5) {
            original_compute = b->iface.graph_compute; b->iface.graph_compute = compute_fault; fail_once = true;
        }
        auto status = ggml_backend_sched_graph_compute_async(sched, graph);
        if (variant == "failure" && step == 5) {
            b->iface.graph_compute = original_compute;
            require(status == GGML_STATUS_FAILED && !fail_once, "injected failure");
            ++result.failures; continue;
        }
        require(status == GGML_STATUS_SUCCESS, "compute graph");
        result.norm.emplace_back(ggml_nelements(norm)); result.base.emplace_back(ggml_nelements(base));
        auto * nb = ggml_backend_sched_get_tensor_backend(sched, norm);
        auto * bb = ggml_backend_sched_get_tensor_backend(sched, base);
        ggml_backend_tensor_get_async(nb, norm, result.norm.back().data(), 0, ggml_nbytes(norm));
        ggml_backend_tensor_get_async(bb, base, result.base.back().data(), 0, ggml_nbytes(base));
        // Host output vectors retain their storage until the scheduler is drained.
    }
    ggml_backend_sched_synchronize(sched);
    for (auto & values : result.norm) require(std::all_of(values.begin(), values.end(), [](float v){return std::isfinite(v);}), "finite norm");
    for (int i = 0; i < 3; ++i) result.sizes.push_back(ggml_backend_sched_get_buffer_size(sched, backends[i]));
    ggml_backend_sched_free(sched);
    for (auto * ctx : contexts) ggml_free(ctx);
    return result;
}
static std::vector<std::vector<float>> capture_test(int mode) {
    setenv("GGML_SCHED_COPY_RESHAPE", mode ? "1" : "0", 1);
    auto * a = ggml_backend_cuda_init(0); auto * b = ggml_backend_cuda_init(1); auto * cpu = ggml_backend_cpu_init();
    ggml_backend_t bs[] = {a,b,cpu};
    ggml_backend_buffer_type_t bt[] = {ggml_backend_get_default_buffer_type(a),ggml_backend_get_default_buffer_type(b),
        ggml_backend_dev_host_buffer_type(ggml_backend_get_device(a))};
    auto * sched = ggml_backend_sched_new(bs,bt,3,2048,true,true);
    require(ggml_backend_sched_set_inputs_async(sched,true), "capture async");
    auto * ctx = ggml_init({1024*1024,nullptr,true}); auto * g = ggml_new_graph_custom(ctx,2048,false);
    const int d = 64, n = 128;
    auto * x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d,n);
    auto * r = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,4,n);
    auto * post = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,n);
    auto * pre = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,n);
    for (auto * t : {x,r,post,pre}) ggml_set_input(t);
    auto * root = ggml_dsv4_hc_post(ctx,x,r,post,nullptr);
    auto * base = ggml_dsv4_hc_pre(ctx,root,pre); ggml_set_output(base);
    auto * v = ggml_reshape_2d(ctx,root,d*4,n);
    auto * norm = ggml_rms_norm(ctx,v,1.0e-5f); ggml_set_output(norm);
    ggml_build_forward_expand(g,base); ggml_build_forward_expand(g,norm);
    for (auto * t : {x,r,post,pre}) ggml_backend_sched_set_tensor_backend(sched,t,cpu);
    ggml_backend_sched_set_tensor_backend(sched,root,a); ggml_backend_sched_set_tensor_backend(sched,v,a);
    ggml_backend_sched_set_tensor_backend(sched,base,b);ggml_backend_sched_set_tensor_backend(sched,norm,b);
    require(ggml_backend_sched_alloc_graph(sched,g), "capture allocate");
    require((norm->src[0]->view_src == base->src[0]) == bool(mode), "capture alias");
    std::vector<std::vector<float>> output;
    std::vector<float> xv(d*n,0), pv(4*n,0), wv(4*n,.25f), rv(d*4*n);
    std::printf("CAPTURE_BEGIN mode=%d\n",mode); std::fflush(stdout);
    for (int step = 0; step < 32; ++step) {
        ggml_backend_sched_prepare_inputs(sched);
        for (size_t i=0; i<rv.size(); ++i) rv[i] = float(step+1)+float(i%19)/32;
        ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);ggml_backend_tensor_set(post,pv.data(),0,pv.size()*4);
        ggml_backend_tensor_set(pre,wv.data(),0,wv.size()*4);ggml_backend_tensor_set(r,rv.data(),0,rv.size()*4);
        require(ggml_backend_sched_graph_compute_async(sched,g)==GGML_STATUS_SUCCESS,"capture compute");
        ggml_backend_sched_synchronize(sched);
        output.emplace_back(ggml_nelements(norm));
        ggml_backend_tensor_get(norm,output.back().data(),0,ggml_nbytes(norm));
    }
    ggml_backend_sched_free(sched);ggml_free(ctx);ggml_backend_free(cpu);ggml_backend_free(b);ggml_backend_free(a);
    std::printf("CAPTURE_END mode=%d\n",mode);std::fflush(stdout);return output;
}
int main(int argc, char ** argv) {
    json report = {{"complete",false},{"cases",json::array()}};
    auto save = [&]{std::ofstream(argv[1]) << report.dump(2) << '\n';};
    try {
        require(argc == 2, "output path");
        ggml_backend_load_all();
        auto * a = ggml_backend_cuda_init(0); auto * b = ggml_backend_cuda_init(1); auto * cpu = ggml_backend_cpu_init();
        require(a && b && cpu, "backends");
        const std::vector<std::string> variants = {"eligible","single_copy","same_device","late_root","different_split", "root_output","root_input","view_output","other_root","other_reader","nested","offset","subrange","strided","cpu_destination","callback","failure"};
        for (const auto & variant : variants) {
            std::vector<std::vector<float>> ref_norm, ref_base;
            const int steps = variant == "eligible" ? 160 : 12;
            for (int mode : {0,1,-1}) {
                auto r = run(a,b,cpu,variant,mode,steps);
                if (mode == 0) {ref_norm = r.norm;ref_base = r.base;}
                else require(r.norm == ref_norm && r.base == ref_base, "OFF/ON/default exact");
                report["cases"].push_back({{"variant",variant},{"mode",mode},{"steps",steps},{"aliases",r.aliases},{"failures",r.failures},{"norm_outputs",r.norm.size()},{"buffers",r.sizes}});
                save(); std::printf("EXACT %s mode=%d aliases=%d failures=%d\n",variant.c_str(),mode,r.aliases,r.failures); std::fflush(stdout);
            }
        }
        ggml_backend_free(cpu);ggml_backend_free(b);ggml_backend_free(a);
        auto capture_off = capture_test(0);
        auto capture_on = capture_test(1);
        require(capture_off == capture_on, "capture exact");
        report["capture_outputs_per_mode"] = capture_on.size();
        report["complete"] = true;save();return 0;
    } catch(const std::exception & e) {report["error"] = e.what();save();std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
