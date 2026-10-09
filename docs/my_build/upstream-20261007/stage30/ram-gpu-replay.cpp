#include "llama.h"
#include "llama-memory-hybrid.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
int main(int argc, char ** argv) {
    try {
        check(argc == 5, "usage: ram-replay model unified n_rs_seq output-directory");
        ggml_backend_load_all();auto mp = llama_model_default_params();mp.n_gpu_layers = 99;
        auto * model = llama_model_load_from_file(argv[1], mp);check(model,"model");
        auto cp = llama_context_default_params();cp.n_ctx = 8192;cp.n_batch = 512;cp.n_ubatch = 64;cp.n_seq_max = 3;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.kv_unified = std::atoi(argv[2]);cp.n_threads = 2;cp.n_threads_batch = 2;cp.n_rs_seq = std::atoi(argv[3]);
        auto * ctx = llama_init_from_model(model,cp);check(ctx,"context");
        auto * mem = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));check(mem && mem->get_mem_idx(),"hybrid indexer");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<int> pos(3,0);std::vector<std::vector<uint8_t>> full(3),partial(3),attn(3),idx(3);
        auto keys = [&](llama_kv_cache * kv, int seq) {
            llama_synchronize(ctx);std::vector<uint8_t> result;
            const auto & cells = kv->get_cells(seq);std::map<int,int> positions;
            for (uint32_t i=0;i<cells.size();++i) if (cells.seq_has(i,seq)) positions.emplace(cells.pos_get(i),i);
            for (uint32_t il : kv->get_layer_ids()) {
                auto * t = kv->get_k_storage(il);size_t row = t->nb[1];
                for (const auto & [p,i] : positions) {
                    const size_t offset = result.size();result.resize(offset+row);
                    ggml_backend_tensor_get(t,result.data()+offset,(kv->get_n_stream()==1?0:seq*t->nb[2])+i*row,row);
                }
            }
            return result;
        };
        auto layout = [&](const char * phase,int seq) {
            const auto & a=mem->get_mem_attn()->get_cells(seq);const auto & b=mem->get_mem_idx()->get_cells(seq);
            int mismatch=0,first=-1;
            for (uint32_t i=0;i<a.size();++i) if (a.seq_has(i,seq)!=b.seq_has(i,seq) || (a.seq_has(i,seq) && a.pos_get(i)!=b.pos_get(i))) {++mismatch;if(first<0)first=i;}
            std::printf("{\"phase\":\"%s\",\"seq\":%d,\"layout_mismatch\":%d,\"first\":%d,\"attn_used\":%u,\"idx_used\":%u}\n",phase,seq,mismatch,first,a.get_used(),b.get_used());std::fflush(stdout);
        };
        auto decode = [&](int seq,int n,FILE * out) {
            auto b=llama_batch_init(n,0,1);b.n_tokens=n;
            for (int i=0;i<n;++i) {int p=pos[seq]++;b.token[i]=(seq*17+p*3+1)%vocab;b.pos[i]=p;b.n_seq_id[i]=1;b.seq_id[i][0]=seq;b.logits[i]=i+1==n;}
            check(llama_decode(ctx,b)==0,"decode");
            if(out)check(std::fwrite(llama_get_logits_ith(ctx,n-1),sizeof(float),vocab,out)==size_t(vocab),"write");
            llama_batch_free(b);
        };
        auto save = [&](int seq,int flags) {
            size_t n=llama_state_seq_get_size_ext(ctx,seq,flags);std::vector<uint8_t> v(n);check(n>0 && llama_state_seq_get_data_ext(ctx,v.data(),n,seq,flags)==n,"save");return v;
        };
        auto load = [&](int seq,int flags,const std::vector<uint8_t> & v) {check(llama_state_seq_set_data_ext(ctx,v.data(),v.size(),seq,flags)==v.size(),"load");};
        int previous = -1;
        for (int phase=0;phase<4;++phase) {
            for (int seq : {2,1,0}) {
                const int prefix=seq==0?432:431;
                const std::string name=std::string(argv[4])+"/phase"+std::to_string(phase)+"-seq"+std::to_string(seq)+".f32";
                if(phase==1 || phase==2) {
                    load(seq,0,full[seq]);layout("full_restore",seq);
                    const bool a=keys(mem->get_mem_attn(),seq)==attn[seq],b=keys(mem->get_mem_idx(),seq)==idx[seq];
                    std::printf("{\"phase\":%d,\"seq\":%d,\"attn_bytes_equal\":%s,\"idx_bytes_equal\":%s}\n",phase,seq,a?"true":"false",b?"true":"false");
                }
                if(previous>=0) {
                    full[previous]=save(previous,0);attn[previous]=keys(mem->get_mem_attn(),previous);idx[previous]=keys(mem->get_mem_idx(),previous);
                    check(llama_memory_seq_rm(llama_get_memory(ctx),previous,0,-1),"clear previous idle");
                }
                if(phase==0 || phase==3) {
                    check(llama_memory_seq_rm(llama_get_memory(ctx),seq,0,-1),"clear");pos[seq]=0;
                    decode(seq,prefix,nullptr);partial[seq]=save(seq,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                } else {
                    load(seq,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,partial[seq]);
                    const auto restored_partial=save(seq,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    std::printf("{\"phase\":%d,\"seq\":%d,\"recurrent_bytes_equal\":%s}\n",phase,seq,restored_partial==partial[seq]?"true":"false");
                    check(llama_memory_seq_rm(llama_get_memory(ctx),seq,prefix,-1),"rewind");pos[seq]=prefix;
                }
                layout("before_suffix",seq);
                FILE * out=std::fopen(name.c_str(),"wb");check(out,"file");decode(seq,4,out);
                for(int i=0;i<63;++i)decode(seq,1,out);
                std::fclose(out);previous=seq;layout("after_decode",seq);
            }
        }
        llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
