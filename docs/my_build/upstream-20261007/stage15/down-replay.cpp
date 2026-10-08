#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<typename T> static std::vector<T> read(const std::string & path, size_t count) {
    std::vector<T> result(count);
    std::ifstream file(path, std::ios::binary);
    check(bool(file.read(reinterpret_cast<char *>(result.data()), count*sizeof(T))), "read failed");
    return result;
}
static void write(const std::string & path, const std::vector<float> & result) {
    std::ofstream file(path, std::ios::binary);
    check(bool(file.write(reinterpret_cast<const char *>(result.data()), result.size()*4)), "write failed");
}
int main(int argc, char ** argv) {
    try {
        check(argc == 3, "usage: down-replay INPUT OUTPUT");
        const std::string root=argv[1], output=argv[2];
        check(!std::filesystem::exists(output), "output exists");
        std::filesystem::create_directories(output);
        auto backend=ggml_backend_cuda_init(0);
        check(backend, "CUDA initialization failed");
        auto wc=ggml_init({2*1024*1024,nullptr,true});
        std::string name,path;
        int type;
        int64_t ne[4];
        uint64_t offset,bytes;
        std::ifstream index(root+"/weight-index.tsv");
        check(bool(index >> name >> type >> ne[0] >> ne[1] >> ne[2] >> ne[3] >> offset >> bytes >> path), "invalid weight index");
        check(name=="blk.0.ffn_down_exps.weight", "unexpected weight");
        auto weight=ggml_new_tensor(wc,static_cast<ggml_type>(type),4,ne);
        check(ggml_nbytes(weight)==bytes, "weight size mismatch");
        auto wb=ggml_backend_alloc_ctx_tensors(wc,backend);
        check(wb, "weight allocation failed");
        ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::ifstream file(path,std::ios::binary);file.seekg(offset);
        std::vector<char> chunk(64*1024*1024);
        for (uint64_t start=0;start<bytes;) {
            const size_t count=std::min<uint64_t>(chunk.size(),bytes-start);
            check(bool(file.read(chunk.data(),count)), "weight read failed");
            ggml_backend_tensor_set(weight,chunk.data(),start,count);start+=count;
        }
        std::ofstream report(output+"/results.jsonl");report<<std::setprecision(17);
        for (int target:{0,86}) for (int source:{1,2,4}) {
            const std::string input=root+"/input"+std::to_string(target)+"-w"+std::to_string(source);
            const auto hidden_data=read<float>(input+"-hidden.f32",2304*6*source);
            const auto ids_data=read<int32_t>(input+"-ids.i32",6*source);
            const auto weights_data=read<float>(input+"-weights.f32",6*source);
            const auto captured=read<float>(input+"-captured.f32",5120*source);
            const int selected=target==86 && source==4 ? 2:0;
            std::vector<float> scalar;
            for (int width:{1,2,4}) for (bool actual:{false,true}) {
                if (actual && (source==1 || width!=source)) continue;
                const std::string label="input"+std::to_string(target)+"-source"+std::to_string(source)+"-w"+std::to_string(width)+(actual?"-actual":"-repeat");
                auto ctx=ggml_init({2*1024*1024,nullptr,true});
                auto graph=ggml_new_graph(ctx);
                auto hidden=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2304,6,width);
                auto all_ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,384,width);
                auto ids=ggml_view_2d(ctx,all_ids,6,width,all_ids->nb[1],0);
                auto weights=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,6,width);
                auto down=ggml_mul_mat_id(ctx,weight,hidden,ids);
                auto weighted=ggml_mul(ctx,down,weights);
                ggml_build_forward_expand(graph,weighted);
                std::vector<ggml_tensor *> views;
                for (int i=0;i<6;++i) {
                    auto view=ggml_view_2d(ctx,weighted,5120,width,weighted->nb[2],i*weighted->nb[1]);
                    ggml_build_forward_expand(graph,view);views.push_back(view);
                }
                auto result=views[0];
                for (int i=1;i<6;++i) {result=ggml_add(ctx,result,views[i]);ggml_build_forward_expand(graph,result);}
                auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"graph allocation failed");
                for (int col=0;col<width;++col) {
                    const int src=actual?col:selected;
                    ggml_backend_tensor_set(hidden,hidden_data.data()+src*2304*6,col*2304*6*4,2304*6*4);
                    ggml_backend_tensor_set(all_ids,ids_data.data()+src*6,col*all_ids->nb[1],6*4);
                    ggml_backend_tensor_set(weights,weights_data.data()+src*6,col*6*4,6*4);
                }
                check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
                std::vector<float> values(5120*width),repeat(values.size());
                ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
                check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"repeat failed");
                ggml_backend_tensor_get(result,repeat.data(),0,repeat.size()*4);
                check(values==repeat,"repeat changed");
                if (width==1) scalar=values;
                double maximum=0,capture_max=0,column_max=0;
                for (int i=0;i<5120;++i) maximum=std::max(maximum,std::abs(double(values[(actual?selected:0)*5120+i])-scalar[i]));
                for (int col=0;col<width;++col) for (int i=0;i<5120;++i) {
                    const int src=actual?col:selected;
                    capture_max=std::max(capture_max,std::abs(double(values[col*5120+i])-captured[src*5120+i]));
                    if (!actual) column_max=std::max(column_max,std::abs(double(values[col*5120+i])-values[i]));
                }
                write(output+"/"+label+".f32",values);
                report<<"{\"label\":\""<<label<<"\",\"target\":"<<target<<",\"source\":"<<source<<",\"width\":"<<width
                      <<",\"actual\":"<<(actual?"true":"false")<<",\"selected_column\":"<<selected<<",\"max_vs_scalar\":"<<maximum
                      <<",\"max_vs_capture\":"<<capture_max<<",\"column_max\":"<<column_max<<",\"repeat_stable\":true}\n";report.flush();
                ggml_backend_buffer_free(buffer);ggml_free(ctx);
            }
        }
        std::ifstream maps("/proc/self/maps");std::ofstream loaded(output+"/loaded-libraries.txt");
        for (std::string line;std::getline(maps,line);) if (line.find("libggml")!=std::string::npos) loaded<<line<<'\n';
        ggml_backend_buffer_free(wb);ggml_free(wc);ggml_backend_free(backend);
    } catch (const std::exception & error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
