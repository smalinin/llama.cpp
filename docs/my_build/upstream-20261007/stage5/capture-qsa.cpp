#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Capture { std::string directory; int step=0, event=0; FILE * manifest; };
static bool callback(ggml_tensor * t, bool ask, void * data) {
    const bool wanted=std::strncmp(t->name,"indexer_score_blk-",18)==0 ||
        std::strncmp(t->name,"indexer_top_k_blk-",18)==0 ||
        std::strncmp(t->name,"indexer_q-",10)==0 || std::strncmp(t->name,"indexer_k-",10)==0;
    if (ask) return wanted;
    if (!wanted) return true;
    auto & c=*static_cast<Capture *>(data);
    const auto file="tensor-"+std::to_string(c.event++)+".bin";
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    std::ofstream(c.directory+"/"+file,std::ios::binary).write(bytes.data(),bytes.size());
    std::fprintf(c.manifest,"{\"file\":\"%s\",\"name\":\"%s\",\"step\":%d,\"type\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"bytes\":%zu}\n",
        file.c_str(),t->name,c.step,t->type,(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],bytes.size());
    std::fflush(c.manifest);return true;
}
static void require(bool ok,const char * msg) { if (!ok) throw std::runtime_error(msg); }
int main(int argc,char ** argv) {
    require(argc==5,"usage: capture-qsa MODEL OUTPUT SLOTS PROMPT");
    const int slots=std::stoi(argv[3]);
    std::filesystem::create_directories(argv[2]);
    Capture capture{argv[2],0,0,std::fopen((std::string(argv[2])+"/tensors.jsonl").c_str(),"w")};
    llama_backend_init();
    auto mp=llama_model_default_params();
    std::vector<float> split(llama_max_devices(),0.0f);
    for (int i=0;i<6;++i) split[i]=i==5 ? 0.4f : 1.0f;
    mp.n_gpu_layers=99;mp.split_mode=LLAMA_SPLIT_MODE_LAYER;mp.tensor_split=split.data();
    auto model=llama_model_load_from_file(argv[1],mp);require(model,"model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=16384;cp.n_batch=256;cp.n_ubatch=128;
    cp.n_seq_max=slots;cp.n_threads=12;cp.n_threads_batch=12;cp.kv_unified=false;
    cp.cb_eval=callback;cp.cb_eval_user_data=&capture;
    auto ctx=llama_init_from_model(model,cp);require(ctx,"context init failed");
    {
        std::ifstream maps("/proc/self/maps");
        std::ofstream loaded(std::string(argv[2])+"/loaded-libraries.txt");
        for (std::string line;std::getline(maps,line);) {
            if (line.find("/libllama.so")!=std::string::npos || line.find("/libggml")!=std::string::npos) loaded<<line<<'\n';
        }
    }
    std::ifstream in(argv[4]);std::string prompt((std::istreambuf_iterator<char>(in)),{});
    auto vocab=llama_model_get_vocab(model);int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
    std::vector<llama_token> tokens(n);require(llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),n,true,true)==n,"tokenize failed");
    auto batch=llama_batch_init(256,0,1);
    for (int start=0;start<n;start+=128/slots) {
        batch.n_tokens=0;
        for (int s=0;s<slots;++s) for (int pos=start;pos<std::min(n,start+128/slots);++pos) {
            const int i=batch.n_tokens++;batch.token[i]=tokens[pos];batch.pos[i]=pos;batch.n_seq_id[i]=1;
            batch.seq_id[i][0]=s;batch.logits[i]=pos==n-1;
        }
        require(llama_decode(ctx,batch)==0,"prefill failed");++capture.step;
    }
    std::ofstream logits_out(std::string(argv[2])+"/logits.bin",std::ios::binary);
    FILE * generated=std::fopen((std::string(argv[2])+"/generated.jsonl").c_str(),"w");
    const int nv=llama_vocab_n_tokens(vocab);
    for (int step=0;step<32;++step) {
        std::vector<llama_token> next(slots);
        for (int s=0;s<slots;++s) {
            const float * logits=llama_get_logits_ith(ctx,-slots+s);require(logits,"missing logits");
            logits_out.write(reinterpret_cast<const char *>(logits),nv*sizeof(float));
            next[s]=std::max_element(logits,logits+nv)-logits;
            std::fprintf(generated,"{\"step\":%d,\"slot\":%d,\"token\":%d}\n",step,s,next[s]);
        }
        if (step==31) break;
        batch.n_tokens=slots;
        for (int s=0;s<slots;++s) {
            batch.token[s]=next[s];batch.pos[s]=n+step;batch.n_seq_id[s]=1;batch.seq_id[s][0]=s;batch.logits[s]=true;
        }
        require(llama_decode(ctx,batch)==0,"decode failed");++capture.step;
    }
    std::fclose(generated);std::fclose(capture.manifest);
    std::printf("captured %d QSA tensors, %d prompt tokens, %d slots, vocab %d\n",capture.event,n,slots,nv);
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
}
