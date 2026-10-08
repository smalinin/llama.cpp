#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
template<typename T> static std::vector<T> read(const std::string & path, size_t count) {
    std::vector<T> out(count);
    std::ifstream f(path, std::ios::binary);
    check(bool(f.read(reinterpret_cast<char *>(out.data()), count*sizeof(T))), "cannot read input");
    return out;
}
static void write(const std::string & path, const std::vector<float> & a) {
    std::ofstream f(path, std::ios::binary);
    check(bool(f.write(reinterpret_cast<const char *>(a.data()), a.size()*4)), "cannot write output");
}
static void set_columns(ggml_tensor * t, const std::vector<float> & a, size_t per_token, int width, bool actual, int col) {
    check(ggml_nbytes(t) == per_token*width*4, "input shape mismatch");
    if (actual) ggml_backend_tensor_set(t, a.data(), 0, per_token*width*4);
    else for (int c = 0; c < width; ++c) ggml_backend_tensor_set(t, a.data()+size_t(col)*per_token, c*per_token*4, per_token*4);
}
int main(int argc, char ** argv) {
    check(argc == 4, "usage: isolated-replay INPUT_DIR projection|post OUTPUT_DIR");
    const std::string root = argv[1], mode = argv[2], out = argv[3];
    check(mode == "projection" || mode == "post", "unknown mode");
    check(!std::filesystem::exists(out), "output exists");
    std::filesystem::create_directories(out);
    auto backend = ggml_backend_cuda_init(0);
    check(backend, "CUDA backend failed");
    auto wc = ggml_init({1024*1024, nullptr, true});
    ggml_tensor * w = nullptr;
    ggml_backend_buffer_t wb = nullptr;
    int n = 5120, m = 5120*4;
    int selected[3];
    std::ifstream selection(root+"/selected-columns.txt");
    check(bool(selection >> selected[0] >> selected[1] >> selected[2]),"missing columns");
    std::vector<float> weights_float;
    if (mode == "projection") {
        int kind;
        std::ifstream config(root+"/config.txt");
        check(bool(config >> kind >> n >> m), "missing projection config");
        auto type = static_cast<ggml_type>(kind);
        check(type == GGML_TYPE_F32 || type == GGML_TYPE_BF16, "unexpected weight type");
        auto weights = read<char>(root+"/weight.bin", ggml_row_size(type,n)*m);
        weights_float.resize(size_t(n)*m);
        if (type == GGML_TYPE_F32) std::memcpy(weights_float.data(), weights.data(), weights.size());
        else ggml_get_type_traits(type)->to_float(weights.data(),weights_float.data(),weights_float.size());
        w = ggml_new_tensor_2d(wc,type,n,m);
        wb = ggml_backend_alloc_ctx_tensors(wc,backend);
        check(wb, "weight allocation failed");
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(w,weights.data(),0,weights.size());
    }
    std::ofstream report(out+"/results.jsonl");
    report << std::setprecision(17);
    for (int source : {1,2,4}) {
        const int col = selected[source==1 ? 0 : source==2 ? 1 : 2];
        std::vector<std::vector<float>> inputs;
        if (mode == "projection") {
            inputs.push_back(read<float>(root+"/input-w"+std::to_string(source)+".f32",size_t(n)*source));
            std::vector<float> cpu(m);
            for (int r = 0; r < m; ++r) {
                double sum = 0;
                for (int j = 0; j < n; ++j) sum += double(weights_float[size_t(r)*n+j])*inputs[0][size_t(col)*n+j];
                cpu[r] = float(sum);
            }
            write(out+"/cpu-source"+std::to_string(source)+".f32",cpu);
        } else {
            const size_t counts[] = {5120,5120*4,4,16};
            for (int k = 0; k < 4; ++k) inputs.push_back(read<float>(root+"/w"+std::to_string(source)+"-src"+std::to_string(k)+".f32",counts[k]*source));
        }
        std::vector<float> scalar;
        for (int width : {1,2,4}) for (bool actual : {false,true}) {
            if (actual && (source == 1 || width != source)) continue;
            auto ctx = ggml_init({1024*1024,nullptr,true});
            auto x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,n,width);
            ggml_tensor * residual = nullptr, * post = nullptr, * comb = nullptr, * y = nullptr;
            if (mode == "projection") {
                y = ggml_mul_mat(ctx,w,x);
                ggml_mul_mat_set_prec(y,GGML_PREC_F32);
            } else {
                residual = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,5120,4,width);
                post = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,width);
                comb = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4,4,width);
                y = ggml_dsv4_hc_post(ctx,x,residual,post,comb);
            }
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph,y);
            check(ggml_graph_n_nodes(graph) == 1, "unexpected graph");
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
            check(buffer, "input allocation failed");
            set_columns(x,inputs[0],n,width,actual,col);
            if (mode == "post") {
                set_columns(residual,inputs[1],5120*4,width,actual,col);
                set_columns(post,inputs[2],4,width,actual,col);
                set_columns(comb,inputs[3],16,width,actual,col);
            }
            check(ggml_backend_graph_compute(backend,graph) == GGML_STATUS_SUCCESS, "compute failed");
            std::vector<float> result(size_t(m)*width), repeat(result.size());
            ggml_backend_tensor_get(y,result.data(),0,result.size()*4);
            check(ggml_backend_graph_compute(backend,graph) == GGML_STATUS_SUCCESS, "repeat failed");
            ggml_backend_tensor_get(y,repeat.data(),0,repeat.size()*4);
            check(std::memcmp(result.data(),repeat.data(),result.size()*4) == 0, "repeat changed");
            if (width == 1) scalar = result;
            double delta = 0, columns = 0;
            for (int r = 0; r < m; ++r) {
                delta = std::max(delta,std::abs(double(result[size_t(actual ? col : 0)*m+r])-scalar[r]));
                if (!actual) for (int c = 1; c < width; ++c) columns = std::max(columns,std::abs(double(result[size_t(c)*m+r])-result[r]));
            }
            const std::string label = "source"+std::to_string(source)+"-w"+std::to_string(width)+(actual ? "-actual" : "-repeat");
            write(out+"/"+label+".f32",result);
            report << "{\"label\":\"" << label << "\",\"max_vs_scalar\":" << delta << ",\"column_max\":" << columns << ",\"repeat_stable\":true}\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    std::ifstream maps("/proc/self/maps");
    std::ofstream loaded(out+"/loaded-libraries.txt");
    for (std::string line;std::getline(maps,line);) if (line.find("libggml") != std::string::npos) loaded << line << '\n';
    if (wb) ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_backend_free(backend);
}
