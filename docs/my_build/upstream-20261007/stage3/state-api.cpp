#define main state_fault_main
#include "state-fault.cpp"
#undef main
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-dsv4.h"
static std::vector<uint8_t> blob(llama_context * ctx,llama_seq_id seq,llama_state_seq_flags flags) {
    std::vector<uint8_t> data(llama_state_seq_get_size_ext(ctx,seq,flags));
    require(!data.empty(),"empty blob");
    require(llama_state_seq_get_data_ext(ctx,data.data(),data.size(),seq,flags)==data.size(),"save failed");
    return data;
}
static std::vector<uint8_t> compressed_keys(llama_context * ctx) {
    std::vector<uint8_t> result;
    auto * mem=dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(ctx));
    if (!mem) return result;
    for (auto * kv : {mem->get_csa(),mem->get_hca(),mem->get_lid()}) {
        for (uint32_t il : kv->get_layer_ids()) {
            auto * tensor=kv->get_k_storage(il);
            const size_t offset=result.size();result.resize(offset+ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor,result.data()+offset,0,ggml_nbytes(tensor));
        }
    }
    return result;
}
int main(int argc,char ** argv) {
    try {
        require(argc>=2,"fixture directory required");ggml_backend_load_all();int passed=0;
        for (const char * fixture : {"llama-dense","glm5next-moe","qwen4exp-moe","glm-dsa-moe","deepseek41-moe"}) {
            if (argc>=3 && std::string(fixture)!=argv[2])continue;
            auto mp=llama_model_default_params();mp.n_gpu_layers=0;
            const std::string path=std::string(argv[1])+"/"+fixture+".gguf";
            llama_model_ptr model(llama_model_load_from_file(path.c_str(),mp));require(bool(model),"model failed");
            for (bool unified : {false,true}) {
                auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=32;cp.n_ubatch=8;cp.n_seq_max=3;
                cp.n_threads=cp.n_threads_batch=2;cp.kv_unified=unified;cp.n_rs_seq=3;
                cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
                llama_context_ptr ctx(llama_init_from_model(model.get(),cp));require(bool(ctx),"context failed");
                for (llama_state_seq_flags flags : {0u,1u,2u,3u}) {
                    llama_memory_clear(llama_get_memory(ctx.get()),true);
                    decode(ctx.get(),0,0,16);decode(ctx.get(),1,0,8);
                    auto preserved=blob(ctx.get(),1,0);auto initial=blob(ctx.get(),0,0);
                    auto saved=blob(ctx.get(),0,flags);
                    auto compressed=compressed_keys(ctx.get());
                    require(llama_state_seq_set_data_ext(ctx.get(),saved.data(),saved.size(),0,flags)==saved.size(),"roundtrip restore failed");
                    require(blob(ctx.get(),0,0)==initial,"roundtrip changed state");
                    require(blob(ctx.get(),1,0)==preserved,"roundtrip changed other seq");
                    require(saved.size()>8,"state header only");
                    const auto nread=llama_state_seq_set_data_ext(ctx.get(),saved.data(),saved.size()-1,0,flags);
                    require(nread==0,"truncated blob accepted");
                    require(blob(ctx.get(),1,0)==preserved,"failed restore changed other seq");
                    if (flags&LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
                        if (auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()))) {
                            require(hybrid->get_mem_attn()->seq_pos_max(0)==15,"partial failure erased attention");
                        }
                        if (auto * dsv4=dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(ctx.get()))) {
                            require(compressed_keys(ctx.get())==compressed,"partial failure erased compressed keys");
                        }
                    } else {
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"failed seq not empty");
                        auto result=decode(ctx.get(),0,0,8);
                        for (float v:result) require(std::isfinite(v),"next request contains NaN");
                    }
                    printf("PASS %s unified=%d flags=%u bytes=%zu\n",fixture,unified,flags,saved.size());++passed;
                }
                for (size_t cut : {size_t(16),size_t(48)}) {
                    llama_memory_clear(llama_get_memory(ctx.get()),true);decode(ctx.get(),0,0,16);decode(ctx.get(),1,0,8);
                    std::vector<uint8_t> full(llama_state_get_size(ctx.get()));
                    require(llama_state_get_data(ctx.get(),full.data(),full.size())==full.size(),"full save failed");
                    require(llama_state_set_data(ctx.get(),full.data(),std::min(cut,full.size()-1))==0,"truncated full state accepted");
                    if (dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(ctx.get())) && cut==16) {
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==15,"early header rejection changed seq0");
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),1)==7,"early header rejection changed seq1");
                    } else {
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"full failure kept seq0");
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),1)==-1,"full failure kept seq1");
                    }
                    printf("PASS full %s unified=%d cut=%zu\n",fixture,unified,cut);++passed;
                }
            }
        }
        printf("SUMMARY passed=%d\n",passed);return 0;
    } catch (const std::exception & e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
