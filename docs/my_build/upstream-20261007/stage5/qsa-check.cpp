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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Spec { int kv, batch, streams; bool pad=false, tie=false; float amplitude=1; };
struct Result { std::vector<float> scores; std::vector<int> selected; size_t bytes; double us=0; };
static std::vector<float> replay_q,replay_k,replay_bias;
static Result run(ggml_backend_t backend, const Spec & s, bool fused, bool reserve_only) {
    auto ctx=ggml_init({16*1024*1024,nullptr,true});
    const int dim=s.pad ? 132 : 128;
    auto qa=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,4,s.batch,s.streams);
    auto ka=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,1,s.kv,s.streams);
    auto bias=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,s.kv,s.batch,s.streams);
    for (auto t : {qa,ka,bias}) ggml_set_input(t);
    auto q=s.pad ? ggml_view_4d(ctx,qa,128,4,s.batch,s.streams,qa->nb[1],qa->nb[2],qa->nb[3],0) : qa;
    auto k=s.pad ? ggml_view_4d(ctx,ka,128,1,s.kv,s.streams,ka->nb[1],ka->nb[2],ka->nb[3],0) : ka;
    ggml_tensor * score;
    if (fused) {
        auto weights=ggml_fill(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,4,s.batch,1,s.streams),1.0f);
        auto mask=ggml_cast(ctx,ggml_scale(ctx,bias,0.0f),GGML_TYPE_F16);
        mask=ggml_reshape_4d(ctx,mask,s.kv,s.batch,1,s.streams);
        score=ggml_lightning_indexer(ctx,q,k,weights,mask);
        score=ggml_reshape_3d(ctx,score,s.kv,s.batch,s.streams);
    } else {
        auto keys=ggml_reshape_3d(ctx,ggml_cont(ctx,k),128,s.kv,s.streams);
        auto queries=ggml_reshape_3d(ctx,ggml_cont(ctx,q),128,4*s.batch,s.streams);
        auto heads=ggml_reshape_4d(ctx,ggml_mul_mat(ctx,keys,queries),s.kv,4,s.batch,s.streams);
        heads=ggml_relu(ctx,heads);
        score=nullptr;
        for (int h=0;h<4;++h) {
            auto slice=ggml_view_3d(ctx,heads,s.kv,s.batch,s.streams,heads->nb[2],heads->nb[3],h*heads->nb[1]);
            score=score ? ggml_add(ctx,score,slice) : ggml_cont(ctx,slice);
        }
    }
    score=ggml_add(ctx,score,bias);
    auto selected=ggml_cont(ctx,ggml_argsort_top_k(ctx,score,std::min(s.kv,32)));
    ggml_set_output(score);ggml_set_output(selected);
    auto graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,score);ggml_build_forward_expand(graph,selected);
    auto ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga,graph)) throw std::runtime_error("allocation failed");
    Result result;result.bytes=ggml_gallocr_get_buffer_size(ga,0);
    if (!reserve_only) {
        for (auto t : {qa,ka,bias}) {
            std::vector<float> data(ggml_nelements(t));
            if (!replay_q.empty()) {
                data=t==qa ? replay_q : t==ka ? replay_k : replay_bias;
                if (data.size()!=size_t(ggml_nelements(t))) throw std::runtime_error("replay shape mismatch");
            } else for (size_t i=0;i<data.size();++i) {
                if (t==bias) data[i]=(i%s.kv < size_t(s.kv/4)) ? -1e9f : 0.0f;
                else data[i]=s.tie && t==qa ? 0.0f : s.amplitude*(std::sin(float(i%10007)*0.071f)+0.3f*std::cos(float(i%701)*0.113f));
            }
            ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));
        }
        auto started=std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
        ggml_backend_synchronize(backend);
        result.us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        result.scores.resize(ggml_nelements(score));result.selected.resize(ggml_nelements(selected));
        ggml_backend_tensor_get(score,result.scores.data(),0,result.scores.size()*sizeof(float));
        ggml_backend_tensor_get(selected,result.selected.data(),0,result.selected.size()*sizeof(int));
    }
    ggml_gallocr_free(ga);ggml_free(ctx);return result;
}
int main(int argc,char ** argv) {
    auto cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,8);
    auto gpu=ggml_backend_cuda_init(0);
    if (!gpu) return 2;
    const bool reserve=argc>1 && std::string(argv[1])=="reserve";
    const bool replay=argc>1 && std::string(argv[1])=="replay";
    std::vector<Spec> specs;
    if (replay) {
        if (argc!=8) throw std::runtime_error("replay Q K SCORES KV BATCH STREAMS");
        auto load=[](const char * path) {
            std::ifstream in(path,std::ios::binary|std::ios::ate);const size_t bytes=in.tellg();in.seekg(0);
            std::vector<float> values(bytes/sizeof(float));in.read(reinterpret_cast<char *>(values.data()),bytes);return values;
        };
        replay_q=load(argv[2]);replay_k=load(argv[3]);replay_bias=load(argv[4]);
        for (auto & value : replay_bias) value=value<-1e8f ? -1e9f : 0.0f;
        specs.push_back({std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7])});
    } else if (reserve) {
        for (int kv : {2048,16384,65536}) for (int batch : {128,512}) for (int streams : {1,4}) specs.push_back({kv,batch,streams});
    } else {
        for (int kv : {1,7,31,32,33,63,64,65,257,1025}) for (int batch : {1,3,7,8,9,17})
        for (int streams : {1,4}) for (bool pad : {false,true}) for (bool tie : {false,true}) specs.push_back({kv,batch,streams,pad,tie});
        specs.push_back({65,9,1,false,false,512});
    }
    int failed=0;
    for (const auto & s : specs) {
        auto before=run(gpu,s,false,reserve);auto after=run(gpu,s,true,reserve);
        double nmse=0,maxabs=0,before_nmse=0;size_t changed=0,sets_changed=0;bool passed=true;
        if (!reserve) {
            auto reference=run(cpu,s,false,false);
            double err=0,norm=0,before_err=0;
            for (size_t i=0;i<after.scores.size();++i) {
                if (!std::isfinite(after.scores[i])) passed=false;
                if (reference.scores[i] < -1e8f) {
                    const double tolerance=4*std::numeric_limits<float>::epsilon()*std::abs(reference.scores[i]);
                    if (std::abs(double(after.scores[i])-reference.scores[i])>tolerance) passed=false;
                    continue;
                }
                const double d=double(after.scores[i])-reference.scores[i];err+=d*d;norm+=double(reference.scores[i])*reference.scores[i];maxabs=std::max(maxabs,std::abs(d));
                const double before_delta=double(before.scores[i])-reference.scores[i];before_err+=before_delta*before_delta;
            }
            nmse=err/std::max(norm,1e-30);before_nmse=before_err/std::max(norm,1e-30);passed=passed && nmse<1e-10;
            for (size_t i=0;i<after.selected.size();++i) changed+=after.selected[i]!=before.selected[i];
            const int row=std::min(s.kv,32);
            for (size_t i=0;i<after.selected.size();i+=row) {
                std::vector<int> a(after.selected.begin()+i,after.selected.begin()+i+row);
                std::vector<int> b(before.selected.begin()+i,before.selected.begin()+i+row);
                std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());sets_changed+=a!=b;
            }
            if (s.tie && changed) passed=false;
        }
        failed+=!passed;
        std::printf("{\"kv\":%d,\"batch\":%d,\"streams\":%d,\"pad\":%s,\"tie\":%s,\"amplitude\":%g,\"passed\":%s,\"nmse\":%.12g,\"before_nmse\":%.12g,\"max_abs\":%.12g,\"selection_changes\":%zu,\"selection_sets_changed\":%zu,\"before_bytes\":%zu,\"after_bytes\":%zu,\"before_us\":%.9g,\"after_us\":%.9g}\n",
            s.kv,s.batch,s.streams,s.pad?"true":"false",s.tie?"true":"false",s.amplitude,passed?"true":"false",nmse,before_nmse,maxabs,changed,sets_changed,before.bytes,after.bytes,before.us,after.us);
        std::fflush(stdout);
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);return failed?1:0;
}
