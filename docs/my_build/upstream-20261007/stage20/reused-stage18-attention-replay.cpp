#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
template<typename T> static std::vector<T> read(const std::string & path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path,std::ios::binary);
    check(bool(f.read(reinterpret_cast<char *>(v.data()),n*sizeof(T))),"input read failed");
    return v;
}
int main(int argc, char ** argv) {
    check(argc==3,"usage: attention-replay INPUT_ROOT OUTPUT");
    const std::string output=argv[2];
    check(!std::filesystem::exists(output),"output exists");
    std::filesystem::create_directories(output);
    auto backend=ggml_backend_cuda_init(0);
    check(backend,"CUDA init failed");
    std::vector<std::filesystem::path> cases;
    for (auto & entry:std::filesystem::directory_iterator(argv[1])) if (entry.is_directory()) cases.push_back(entry.path());
    std::sort(cases.begin(),cases.end());
    std::ofstream report(output+"/results.jsonl");
    for (auto & entry:cases) {
        const std::string root=entry.string(),label=entry.filename().string();
        int d,heads,nk;
        std::ifstream config(root+"/shape.txt");config>>d>>heads>>nk;
        check(config.good() && d==512 && heads==64 && nk>0,"unsupported shape");
        const auto params=read<int32_t>(root+"/op-params.bin",16);
        float scale,bias,cap;
        std::memcpy(&scale,params.data(),4);std::memcpy(&bias,params.data()+1,4);std::memcpy(&cap,params.data()+2,4);
        auto ctx=ggml_init({4*1024*1024,nullptr,true});
        auto qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,heads,1);
        auto q=ggml_permute(ctx,qb,0,2,1,3);
        auto kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,d,1,nk);
        auto k=ggml_permute(ctx,kb,0,2,1,3);
        auto vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,d,1,nk);
        auto v=ggml_permute(ctx,vb,0,2,1,3);
        auto m=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,nk,1);
        auto s=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,heads);
        auto y=ggml_flash_attn_ext(ctx,q,k,v,m,scale,bias,cap);
        ggml_flash_attn_ext_add_sinks(y,s);
        ggml_flash_attn_ext_set_prec(y,ggml_prec(params[3]));
        ggml_flash_attn_ext_set_n_kv_max(y,params[4]);
        auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"allocation failed");
        const auto query=read<float>(root+"/q.f32",d*heads);
        const auto keys=read<ggml_fp16_t>(root+"/k.f16",d*nk);
        const auto values=read<ggml_fp16_t>(root+"/v.f16",d*nk);
        const auto mask=read<ggml_fp16_t>(root+"/mask.f16",nk);
        const auto sinks=read<float>(root+"/sinks.f32",heads);
        ggml_backend_tensor_set(qb,query.data(),0,query.size()*4);
        ggml_backend_tensor_set(kb,keys.data(),0,keys.size()*2);
        ggml_backend_tensor_set(vb,values.data(),0,values.size()*2);
        ggml_backend_tensor_set(m,mask.data(),0,mask.size()*2);
        ggml_backend_tensor_set(s,sinks.data(),0,sinks.size()*4);
        std::vector<float> out(d*heads),repeat(out.size());
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
        ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat failed");
        ggml_backend_tensor_get(y,repeat.data(),0,repeat.size()*4);
        check(std::memcmp(out.data(),repeat.data(),out.size()*4)==0,"repeat differs");
        std::ofstream(output+"/"+label+".f32",std::ios::binary).write(reinterpret_cast<char *>(out.data()),out.size()*4);
        report<<"{\"case\":\""<<label<<"\",\"nk\":"<<nk<<",\"repeat_bit_exact\":true}\n";report.flush();
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
