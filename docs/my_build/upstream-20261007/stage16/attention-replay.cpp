#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

extern "C" void attention_reference(const float *, const float *, const float *, const float *, const float *, int, int, int, int, float *);

static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }

template<typename T> static std::vector<T> read(const std::string & path, size_t n) {
    std::vector<T> values(n);
    std::ifstream f(path, std::ios::binary);
    check(bool(f.read(reinterpret_cast<char *>(values.data()), n*sizeof(T))), "input read failed");
    return values;
}

int main(int argc, char ** argv) {
    check(argc == 3, "usage: attention-replay INPUT_DIR OUTPUT_DIR");
    const std::string root = argv[1], output = argv[2];
    check(!std::filesystem::exists(output), "output exists");
    std::filesystem::create_directories(output);
    constexpr int d=512, heads=64, nk=1792;
    const auto query=read<float>(root+"/q.f32", d*heads);
    const auto keys=read<ggml_fp16_t>(root+"/k.f16", d*nk);
    const auto values=read<ggml_fp16_t>(root+"/v.f16", d*nk);
    const auto mask=read<ggml_fp16_t>(root+"/mask.f16", nk);
    const auto sinks=read<float>(root+"/sinks.f32", heads);
    const auto captured=read<float>(root+"/out.f32", d*heads);
    auto backend=ggml_backend_cuda_init(0);
    check(backend, "CUDA init failed");
    std::vector<float> scalar;
    std::ofstream report(output+"/results.jsonl");
    for (int width : {1}) {
        auto ctx=ggml_init({16*1024*1024,nullptr,true});
        auto qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,heads,width);
        auto q=ggml_permute(ctx,qb,0,2,1,3);
        auto kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,d,1,nk);
        auto k=ggml_permute(ctx,kb,0,2,1,3);
        auto vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,d,1,nk);
        auto v=ggml_permute(ctx,vb,0,2,1,3);
        auto m=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,nk,width);
        auto s=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,heads);
        auto y=ggml_flash_attn_ext(ctx,q,k,v,m,1.0f/std::sqrt(float(d)),0,0);
        ggml_flash_attn_ext_add_sinks(y,s);
        ggml_flash_attn_ext_set_prec(y,GGML_PREC_F32);
        ggml_flash_attn_ext_set_n_kv_max(y,1152);
        auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        check(buffer,"buffer failed");
        ggml_backend_tensor_set(kb,keys.data(),0,keys.size()*2);
        ggml_backend_tensor_set(vb,values.data(),0,values.size()*2);
        ggml_backend_tensor_set(s,sinks.data(),0,sinks.size()*4);
        for (int c=0;c<width;++c) {
            ggml_backend_tensor_set(qb,query.data(),c*query.size()*4,query.size()*4);
            ggml_backend_tensor_set(m,mask.data(),c*mask.size()*2,mask.size()*2);
        }
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"attention failed");
        std::vector<float> actual(d*heads*width), repeat(actual.size());
        ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat failed");
        ggml_backend_tensor_get(y,repeat.data(),0,repeat.size()*4);
        check(actual==repeat,"repeat differs");
        if (width==1) scalar=actual;
        double maximum=0,squares=0,column_max=0,capture_max=0;
        for (int i=0;i<d*heads;++i) {
            double delta=double(actual[i])-scalar[i];maximum=std::max(maximum,std::abs(delta));squares+=delta*delta;
            capture_max=std::max(capture_max,std::abs(double(actual[i])-captured[i]));
            for (int c=1;c<width;++c) column_max=std::max(column_max,std::abs(double(actual[i])-actual[c*d*heads+i]));
        }
        report<<"{\"width\":"<<width<<",\"max_abs_vs_scalar\":"<<maximum<<",\"rms_vs_scalar\":"<<std::sqrt(squares/(d*heads))
              <<",\"max_abs_vs_capture\":"<<capture_max<<",\"identical_column_max\":"<<column_max<<",\"repeat_stable\":true}\n";
        report.flush();
        std::ofstream(output+"/w"+std::to_string(width)+".f32",std::ios::binary).write(reinterpret_cast<char *>(actual.data()),actual.size()*4);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    std::ifstream maps("/proc/self/maps");
    std::ofstream loaded(output+"/loaded-libraries.txt");
    for (std::string line; std::getline(maps,line);) if (line.find("libggml") != std::string::npos) loaded << line << char(10);
    ggml_backend_free(backend);
}
