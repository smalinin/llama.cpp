#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool ok,const char * what) { if (!ok) throw std::runtime_error(what); }
static void decode(llama_context * ctx) {
    llama_token tokens[]={1,4,7,10,13,16,19,22};
    check(llama_decode(ctx,llama_batch_get_one(tokens,8))==0,"decode failed");
    check(llama_get_logits_ith(ctx,-1)!=nullptr,"no logits");
}
static std::vector<uint8_t> read(const std::string & path) {
    std::ifstream f(path,std::ios::binary);check(bool(f),"file missing");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),{});
}
static void write(const std::string & path,const std::vector<uint8_t> & bytes) {
    std::ofstream f(path,std::ios::binary);f.write((const char*)bytes.data(),bytes.size());check(bool(f),"write failed");
}
int main(int argc,char ** argv) {
    try {
        check(argc==4,"usage: fixture prefix save|match|reject");
        ggml_backend_load_all();
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        llama_model_ptr model(llama_model_load_from_file(argv[1],mp));check(bool(model),"model failed");
        const std::string mode=argv[3];int passed=0;
        for (ggml_type kv : {GGML_TYPE_F16,GGML_TYPE_Q8_0}) {
            auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_seq_max=2;cp.n_batch=16;cp.n_ubatch=8;
            cp.n_threads=cp.n_threads_batch=2;cp.kv_unified=true;cp.type_k=cp.type_v=kv;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            llama_context_ptr ctx(llama_init_from_model(model.get(),cp));check(bool(ctx),"context failed");
            const std::string prefix=std::string(argv[2])+"."+ggml_type_name(kv);
            llama_token tokens[]={1,4,7,10,13,16,19,22};
            if (mode=="save") {
                decode(ctx.get());
                check(llama_state_save_file(ctx.get(),(prefix+".session.bin").c_str(),tokens,8),"save session failed");
                check(llama_state_seq_save_file(ctx.get(),(prefix+".seq.bin").c_str(),0,tokens,8)>0,"save sequence failed");
                std::vector<uint8_t> full(llama_state_get_size(ctx.get()));
                check(llama_state_get_data(ctx.get(),full.data(),full.size())==full.size(),"save raw full failed");
                write(prefix+".full.raw",full);
                std::vector<uint8_t> seq(llama_state_seq_get_size(ctx.get(),0));
                check(llama_state_seq_get_data(ctx.get(),seq.data(),seq.size(),0)==seq.size(),"save raw seq failed");
                write(prefix+".seq.raw",seq);
                auto data=read(prefix+".session.bin");uint32_t version;memcpy(&version,data.data()+4,4);
                auto sd=read(prefix+".seq.bin");uint32_t sv;memcpy(&sv,sd.data()+4,4);
                printf("SAVED kv=%s session_version=%u sequence_version=%u full_bytes=%zu seq_bytes=%zu\n",ggml_type_name(kv),version,sv,full.size(),seq.size());
                continue;
            }
            check(mode=="match"||mode=="reject","invalid mode");
            for (const char * kind : {"session","sequence","full-raw","sequence-raw"}) {
                llama_memory_clear(llama_get_memory(ctx.get()),true);
                decode(ctx.get());
                size_t count=0,nread=0;
                if (strcmp(kind,"session")==0) nread=llama_state_load_file(ctx.get(),(prefix+".session.bin").c_str(),tokens,8,&count);
                if (strcmp(kind,"sequence")==0) nread=llama_state_seq_load_file(ctx.get(),(prefix+".seq.bin").c_str(),0,tokens,8,&count);
                if (strcmp(kind,"full-raw")==0) {auto data=read(prefix+".full.raw");nread=llama_state_set_data(ctx.get(),data.data(),data.size());}
                if (strcmp(kind,"sequence-raw")==0) {auto data=read(prefix+".seq.raw");nread=llama_state_seq_set_data(ctx.get(),data.data(),data.size(),0);}
                check(mode=="match"?nread>0:nread==0,"format compatibility expectation failed");
                printf("PASS mode=%s kv=%s kind=%s read=%zu\n",mode.c_str(),ggml_type_name(kv),kind,nread);++passed;
            }
        }
        printf("SUMMARY passed=%d\n",passed);return 0;
    } catch (const std::exception & e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
