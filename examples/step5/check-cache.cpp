#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> decode(llama_context * ctx, int begin, int count, int seq, int salt=0) {
    auto batch = llama_batch_init(count,0,1);
    batch.n_tokens = count;
    for (int i=0;i<count;++i) {
        batch.token[i]=(begin+i+salt)%127;
        batch.pos[i]=begin+i;
        batch.n_seq_id[i]=1; batch.seq_id[i][0]=seq;
        batch.logits[i]=i==count-1;
    }
    if (llama_decode(ctx,batch)!=0) { throw std::runtime_error("decode failed"); }
    auto * p=llama_get_logits_ith(ctx,-1);
    std::vector<float> result(p,p+128);
    llama_batch_free(batch);
    return result;
}

static std::vector<float> prefill(llama_context * ctx, int length, int chunk, int seq=0, int salt=0) {
    std::vector<float> result;
    for (int p=0;p<length;p+=chunk) { result=decode(ctx,p,std::min(chunk,length-p),seq,salt); }
    return result;
}

static void compare(const char * label, const std::vector<float> & a, const std::vector<float> & b) {
    float maximum=0;
    for (int i=0;i<128;++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { throw std::runtime_error("non-finite logits"); }
        maximum=std::max(maximum,std::abs(a[i]-b[i]));
    }
    bool greedy=std::max_element(a.begin(),a.end())-a.begin()==std::max_element(b.begin(),b.end())-b.begin();
    std::printf("%s max_abs=%.9g greedy_equal=%d\n",label,maximum,greedy); std::fflush(stdout);
    if (maximum>3e-4 || !greedy) { throw std::runtime_error(label); }
}

int main(int argc, char ** argv) {
    try {
        setenv("LLAMA_STEP5_EXPERIMENTAL_CSA","1",1);
        llama_backend_init();
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=99; mp.split_mode=LLAMA_SPLIT_MODE_NONE;
        auto * model=llama_model_load_from_file(argc>1?argv[1]:"/tmp/step5-small-csa.gguf",mp);
        if (!model) { throw std::runtime_error("model load"); }
        const bool unified=argc>2 && std::atoi(argv[2])!=0;
        auto context=[&](int chunk) {
            auto cp=llama_context_default_params();
            cp.n_ctx=16384; cp.n_seq_max=2; cp.n_batch=512; cp.n_ubatch=chunk;
            cp.n_threads=cp.n_threads_batch=4; cp.kv_unified=unified;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            auto * ctx=llama_init_from_model(model,cp);
            if (!ctx) { throw std::runtime_error("context init"); }
            return ctx;
        };
        auto * a=context(256);
        auto baseline=prefill(a,4200,256);
        auto * b=context(128);
        compare("chunked_prefill",baseline,prefill(b,4200,128));
        llama_free(b);
        auto mem=llama_get_memory(a);
        auto next=decode(a,4200,1,0);
        if (!llama_memory_seq_rm(mem,0,4200,-1)) { throw std::runtime_error("rollback"); }
        decode(a,4200,4,0,17);
        llama_memory_seq_rm(mem,0,4200,-1);
        compare("verification_rollback",next,decode(a,4200,1,0));
        llama_memory_seq_rm(mem,0,4200,-1);
        llama_memory_seq_cp(mem,0,1,0,-1);
        compare("prefix_copy",next,decode(a,4200,1,1));
        llama_memory_seq_rm(mem,1,0,-1);
        prefill(a,4200,256,1,31);
        compare("sequence_isolation",next,decode(a,4200,1,0));
        llama_memory_seq_rm(mem,0,4200,-1);
        size_t size=llama_state_seq_get_size(a,0);
        std::vector<uint8_t> state(size);
        if (llama_state_seq_get_data(a,state.data(),size,0)!=size) { throw std::runtime_error("save"); }
        b=context(512);
        if (llama_state_seq_set_data(b,state.data(),size,0)!=size) { throw std::runtime_error("restore"); }
        compare("state_restore",next,decode(b,4200,1,0));
        compare("state_original",next,decode(a,4200,1,0));
        for (int p=4201;p<4216;++p) { compare("graph_reuse",decode(a,p,1,0),decode(b,p,1,0)); }
        llama_free(a); llama_free(b); llama_model_free(model); llama_backend_free();
        std::puts("PASS");
    } catch (const std::exception & e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
