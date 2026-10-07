#define main state_compat_main
#include "state-compat.cpp"
#undef main
#include <filesystem>
int main(int argc,char ** argv) {
    if (argc!=3) return 2;
    ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    llama_model_ptr model(llama_model_load_from_file(argv[1],mp));if(!model)return 2;
    auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=16;cp.n_ubatch=8;cp.n_threads=cp.n_threads_batch=2;
    llama_context_ptr ctx(llama_init_from_model(model.get(),cp));if(!ctx)return 2;
    decode(ctx.get());llama_token tokens[]={1,4,7,10,13,16,19,22};
    check(llama_state_seq_save_file(ctx.get(),argv[2],0,tokens,8)>0,"save failed");
    std::filesystem::resize_file(argv[2],std::filesystem::file_size(argv[2])-1);
    size_t n=0;const size_t result=llama_state_seq_load_file(ctx.get(),argv[2],0,tokens,8,&n);
    check(result==0,"truncated state accepted");
    check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"failed sequence not empty");
    decode(ctx.get());printf("PASS last-byte truncation rejected and next decode succeeded\n");return 0;
}
