#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool value,const char * message) {if(!value)throw std::runtime_error(message);}
static std::vector<char> read(const std::string & path,size_t size) {
    std::vector<char> result(size);std::ifstream f(path,std::ios::binary);
    check(bool(f.read(result.data(),size)),"read failed");return result;
}
int main(int argc,char ** argv) {
    check(argc==3,"usage: indexer-replay INPUT_ROOT OUTPUT");const std::string out=argv[2];
    check(!std::filesystem::exists(out),"output exists");std::filesystem::create_directories(out);
    auto backend=ggml_backend_cuda_init(0);check(backend,"backend failed");
    std::vector<std::filesystem::path> cases;
    for(auto & e:std::filesystem::directory_iterator(argv[1]))if(e.is_directory())cases.push_back(e.path());
    std::sort(cases.begin(),cases.end());std::ofstream report(out+"/results.jsonl");
    for(const auto & entry:cases) {
        const auto root=entry.string(),label=entry.filename().string();int width,nk;
        std::ifstream config(root+"/shape.txt");check(bool(config>>width>>nk),"shape failed");
        check((width==1 || width==4) && nk==7424,"unsupported shape");
        auto ctx=ggml_init({1024*1024,nullptr,true});
        auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,32,width);
        auto k=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,128,1,nk);
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,width);
        auto m=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,nk,width);
        auto score=ggml_lightning_indexer(ctx,q,k,w,m);
        auto top=ggml_cont(ctx,ggml_top_k(ctx,score,2048));
        auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,score);ggml_build_forward_expand(graph,top);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"allocation failed");
        for(auto item:std::vector<std::pair<ggml_tensor *,std::string>>{{q,"q.f32"},{k,"k.f16"},{w,"weights.f32"},{m,"mask.f16"}}) {
            auto bytes=read(root+"/"+item.second,ggml_nbytes(item.first));ggml_backend_tensor_set(item.first,bytes.data(),0,bytes.size());
        }
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
        auto get=[](ggml_tensor * t){std::vector<char> data(ggml_nbytes(t));ggml_backend_tensor_get(t,data.data(),0,data.size());return data;};
        const auto scores=get(score),selected=get(top);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat failed");
        check(scores==get(score) && selected==get(top),"repeat differs");
        std::ofstream(out+"/"+label+"-scores.f32",std::ios::binary).write(scores.data(),scores.size());
        std::ofstream(out+"/"+label+"-top.i32",std::ios::binary).write(selected.data(),selected.size());
        report<<"{\"case\":\""<<label<<"\",\"width\":"<<width<<",\"nk\":"<<nk<<",\"repeat_bit_exact\":true}\n";report.flush();
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    std::ifstream maps("/proc/self/maps");std::ofstream loaded(out+"/loaded-libraries.txt");
    for(std::string line;std::getline(maps,line);)if(line.find("libggml")!=std::string::npos)loaded<<line<<'\n';
    ggml_backend_free(backend);
}
