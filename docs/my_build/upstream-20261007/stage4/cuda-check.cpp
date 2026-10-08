#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Spec {
    std::string tag;
    std::string layout="contiguous";
    bool mat=false, view=false, branch=false;
    int k=128, m=7, n=3, heads=2;
    ggml_type type=GGML_TYPE_F32;
    float scale=0.08838835f, bias=0;
};
struct Result { std::vector<float> values; double us=0; };
static Result run(ggml_backend_t backend, const Spec & s, bool perf) {
    ggml_init_params params{16*1024*1024,nullptr,true};
    ggml_context * ctx=ggml_init(params);
    ggml_tensor * a;
    ggml_tensor * b=nullptr;
    ggml_tensor * out;
    if (s.mat) {
        a=s.layout=="pad" ? ggml_new_tensor_3d(ctx,s.type,s.k+4,s.m,s.heads) :
          s.layout=="permute" ? ggml_new_tensor_3d(ctx,s.type,s.k,s.heads,s.m) : ggml_new_tensor_3d(ctx,s.type,s.k,s.m,s.heads);
        b=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,s.k,s.n,s.heads);
        ggml_tensor * matrix=a;
        if (s.layout=="pad") matrix=ggml_view_3d(ctx,a,s.k,s.m,s.heads,a->nb[1],a->nb[2],0);
        if (s.layout=="permute") matrix=ggml_permute(ctx,a,0,2,1,3);
        out=ggml_mul_mat(ctx,matrix,b);
    } else {
        a=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,s.view ? 2*s.k : s.k, s.view ? 2*s.m : s.m,s.n,s.heads);
        ggml_tensor * x=a;
        if (s.view) x=ggml_view_4d(ctx,a,s.k,s.m,s.n,s.heads,a->nb[1],a->nb[2],a->nb[3],sizeof(float));
        ggml_tensor * norm=ggml_rms_norm(ctx,x,1e-6f);
        out=ggml_scale_bias(ctx,norm,s.scale,s.bias);
        if (s.branch) out=ggml_add(ctx,out,norm);
    }
    ggml_set_input(a);
    if (b) ggml_set_input(b);
    ggml_set_output(out);
    ggml_cgraph * graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if (!buffer) throw std::runtime_error("allocation failed");
    for (auto t : {a,b}) {
        if (!t) continue;
        std::vector<float> input(ggml_nelements(t));
        for (size_t i=0;i<input.size();++i) input[i]=std::sin(float(i%997)*0.173f)*0.8f+std::cos(float(i%257)*0.057f)*0.15f;
        if (t->type==GGML_TYPE_F32) ggml_backend_tensor_set(t,input.data(),0,ggml_nbytes(t));
        else {
            std::vector<uint16_t> packed(input.size());
            for (size_t i=0;i<input.size();++i) packed[i]=t->type==GGML_TYPE_F16 ? ggml_fp32_to_fp16(input[i]) : ggml_fp32_to_bf16(input[i]).bits;
            ggml_backend_tensor_set(t,packed.data(),0,ggml_nbytes(t));
        }
    }
    for (int i=0;i<3;++i) if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    Result result;
    if (perf) {
        std::vector<double> times;
        for (int block=0;block<5;++block) {
            ggml_backend_synchronize(backend);
            auto start=std::chrono::steady_clock::now();
            for (int i=0;i<100;++i) if (ggml_backend_graph_compute_async(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
            ggml_backend_synchronize(backend);
            times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100);
        }
        std::sort(times.begin(),times.end()); result.us=times[2];
    }
    result.values.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out,result.values.data(),0,ggml_nbytes(out));
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
    return result;
}
int main(int argc,char ** argv) {
    const std::string mode=argc>1 ? argv[1] : "all";
    auto gpu=ggml_backend_cuda_init(0);auto cpu=ggml_backend_cpu_init();
    if (!gpu || !cpu) return 2;
    ggml_backend_cpu_set_n_threads(cpu,4);
    std::vector<Spec> specs;
    if (mode=="norm-profile" || mode=="mmvf-profile") {
        Spec s;s.tag=mode;s.mat=mode=="mmvf-profile";s.k=128;s.m=s.mat?33:512;s.n=s.mat?4:1;s.heads=1;s.type=s.mat?GGML_TYPE_F16:GGML_TYPE_F32;
        specs.push_back(s);
    } else if (mode=="strides") {
        for (auto type : {GGML_TYPE_F16,GGML_TYPE_BF16}) for (int k : {128,130}) for (int m : {33,512,513})
        for (int n : {1,4,8,9,32}) for (const std::string layout : {"pad","permute"}) {
            Spec s;s.mat=true;s.type=type;s.k=k;s.m=m;s.n=n;s.heads=4;s.layout=layout;
            s.tag=std::string(ggml_type_name(type))+"-k"+std::to_string(k)+"-m"+std::to_string(m)+"-n"+std::to_string(n)+"-"+layout;
            specs.push_back(s);
        }
    } else {
        for (int k : {1,32,128,1023,1024,4096}) for (float scale : {0.0f,-0.125f,0.08838835f})
        for (float bias : {0.0f,0.5f}) for (bool view : {false,true}) for (bool branch : {false,true}) {
            Spec s;s.k=k;s.scale=scale;s.bias=bias;s.view=view;s.branch=branch;
            s.tag="norm-k"+std::to_string(k)+"-s"+std::to_string(scale)+"-b"+std::to_string(bias)+"-v"+std::to_string(view)+"-f"+std::to_string(branch);
            specs.push_back(s);
        }
        for (auto type : {GGML_TYPE_F16,GGML_TYPE_BF16}) for (int k : {128,130}) for (int m : {32,33,511,512,513,1024})
        for (int n : {1,2,3,4,8,9,16,32}) for (int heads : {1,4}) {
            Spec s;s.mat=true;s.type=type;s.k=k;s.m=m;s.n=n;s.heads=heads;
            s.tag=std::string(ggml_type_name(type))+"-k"+std::to_string(k)+"-m"+std::to_string(m)+"-n"+std::to_string(n)+"-h"+std::to_string(heads);
            specs.push_back(s);
        }
    }
    std::ofstream raw(argc>2?argv[2]:"cuda-check.bin",std::ios::binary);
    int failed=0;size_t offset=0;
    for (const auto & s : specs) {
        auto actual=run(gpu,s,mode!="correctness" && mode!="strides");auto expected=run(cpu,s,false);
        double err=0,ref=0,maxerr=0;bool finite=true;
        for (size_t i=0;i<actual.values.size();++i) {
            double d=actual.values[i]-expected.values[i];err+=d*d;ref+=double(expected.values[i])*expected.values[i];maxerr=std::max(maxerr,std::abs(d));
            finite=finite && std::isfinite(actual.values[i]);
        }
        double nmse=err/(ref+1e-30);bool passed=finite && nmse<(s.mat?5e-4:1e-10);
        failed+=!passed;
        std::printf("{\"case\":\"%s\",\"passed\":%s,\"nmse\":%.12g,\"max_abs\":%.12g,\"us\":%.9g,\"offset\":%zu,\"count\":%zu}\n",s.tag.c_str(),passed?"true":"false",nmse,maxerr,actual.us,offset,actual.values.size());
        std::fflush(stdout);
        raw.write(reinterpret_cast<const char *>(actual.values.data()),actual.values.size()*sizeof(float));offset+=actual.values.size()*sizeof(float);
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);
    return failed ? 1 : 0;
}
