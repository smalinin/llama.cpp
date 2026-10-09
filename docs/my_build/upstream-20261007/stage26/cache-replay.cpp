#include "llama-model.h"
#include "general-callback.h"
#include "raw-layout.h"
#include <sstream>

using namespace ds14;
struct EvalState {
    PrecisionOptions options{"decode-scalar-fa-upgate-hc-router-down-compressor-indexer", false};
    DownState down{&options};
    ds22::RawPlan plan;
    llama_context * ctx = nullptr;
    std::map<ggml_tensor *, ggml_prec> saved;
    FILE * trace = nullptr;
    bool measured = false;
    int raw = 0;
};
static bool evaluate(ggml_tensor * tensor, bool ask, void * data) {
    auto & state = *static_cast<EvalState *>(data);
    auto & options = state.options;
    if (ask && options.decoding && tensor->op == GGML_OP_MUL_MAT) {
        const auto type = tensor->src[0]->type;
        if (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) state.saved.emplace(tensor, static_cast<ggml_prec>(tensor->op_params[0]));
    }
    if (!ask && options.decoding && tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        const int layer = std::stoi(std::string(tensor->src[0]->name).substr(2));
        if (layer == 0) {
            state.raw = tensor->src[1]->ne[1];
            ds22::check_raw_plan(llama_get_memory(state.ctx), state.plan, state.raw);
        }
        if (state.measured && (layer == 0 || layer == 2 || layer == 20)) {
            for (int col = 0; col < tensor->src[0]->ne[1]; ++col) {
                const int ratio = options.compress_ratios[layer];
                const int p = options.absolute_position+col;
                const int comp = ratio ? std::max(256, ((p+1)/ratio+255)/256*256) : 0;
                std::fprintf(state.trace,"{\"pos\":%d,\"width\":%lld,\"col\":%d,\"layer\":%d,\"source_raw\":%d,\"raw\":%d,\"source_total\":%lld,\"effective_total\":%d,\"n_kv_max\":%d,\"physical_index\":%u,\"before_used_max\":%d}\n",
                    p,(long long)tensor->src[0]->ne[1],col,layer,state.raw,state.plan.extents[col],
                    (long long)tensor->src[1]->ne[1],state.plan.extents[col]+comp,tensor->op_params[4],state.plan.indices[col],state.plan.before_max);
            }
        }
    }
    return down_precision(tensor,ask,&state.down);
}
static void decode(EvalState & state, llama_batch & batch, bool generation) {
    check(state.saved.empty() && state.down.inputs.empty(),"dirty callback state");
    state.options.decoding = generation;
    state.options.absolute_position = batch.pos[0];
    if (generation) {
        state.plan = ds22::raw_plan(llama_get_memory(state.ctx),batch);
        state.options.raw_query_extents = state.plan.extents;
    } else state.options.raw_query_extents.clear();
    check(llama_decode(state.ctx,batch)==0,"decode failed");
    llama_synchronize(state.ctx);
    for (const auto & item : state.saved) ggml_mul_mat_set_prec(item.first,item.second);
    state.saved.clear();state.down.inputs.clear();state.options.decoding=false;
}
static std::vector<llama_token> tokens(const std::string & file, llama_model * model) {
    if (file.size()>=4 && file.substr(file.size()-4)==".txt") {
        const auto text=read_text(file.c_str());const auto * vocab=llama_model_get_vocab(model);
        const int count=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);std::vector<llama_token> result(count);
        check(llama_tokenize(vocab,text.data(),text.size(),result.data(),count,true,true)==count,"tokenization failed");return result;
    }
    std::ifstream input(file,std::ios::binary|std::ios::ate);check(bool(input),"token file open failed");
    const auto size=input.tellg();check(size>0 && size%4==0,"token file size failed");
    std::vector<llama_token> result(size/4);input.seekg(0);check(bool(input.read(reinterpret_cast<char *>(result.data()),size)),"token read failed");return result;
}
static void replay(llama_model * model,const std::vector<llama_token> & prompt,const std::vector<llama_token> & forced,int width,bool cached,const std::string & directory,const std::vector<llama_token> & warmup) {
    std::filesystem::create_directories(directory);
    EvalState state;
    state.options.compress_ratios.assign(model->hparams.dsv4_compress_ratios.begin(),model->hparams.dsv4_compress_ratios.begin()+model->hparams.n_layer());
    auto cp=llama_context_default_params();cp.n_ctx=8192;cp.n_batch=2048;cp.n_ubatch=512;cp.n_seq_max=1;
    cp.n_threads=cp.n_threads_batch=12;cp.type_k=cp.type_v=GGML_TYPE_F16;
    cp.n_outputs_max=cp.n_outputs_max_per_seq=4;cp.kv_unified=false;cp.swa_full=false;cp.n_rs_seq=3;
    cp.cb_eval=evaluate;cp.cb_eval_user_data=&state;
    state.ctx=llama_init_from_model(model,cp);check(state.ctx,"context failed");
    auto * memory=llama_get_memory(state.ctx);llama_memory_clear(memory,true);
    auto batch=llama_batch_init(2048,0,1);
    auto fill=[&](const std::vector<llama_token> & source,int start,int count,int pos,bool outputs) {
        batch.n_tokens=count;
        for (int i=0;i<count;++i) {batch.token[i]=source[start+i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=outputs || i+1==count;}
    };
    auto prefill=[&](int start,int end) {
        while(start<end) {const int n=std::min(2048,end-start);fill(prompt,start,n,start,false);decode(state,batch,false);start+=n;}
    };
    const int checkpoint=prompt.size()-4;
    const int early=std::max(0,(int)prompt.size()-516);prefill(0,early);prefill(early,checkpoint);
    std::vector<uint8_t> saved_state;
    if(cached) {
        const size_t size=llama_state_seq_get_size_ext(state.ctx,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);saved_state.resize(size);
        check(llama_state_seq_get_data_ext(state.ctx,saved_state.data(),size,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)==size,"checkpoint save failed");
        std::ofstream(directory+"/checkpoint.state",std::ios::binary).write(reinterpret_cast<const char *>(saved_state.data()),size);
    }
    prefill(checkpoint,prompt.size());
    if(cached) {
        for(int start=0;start<(int)warmup.size()-1;++start) {fill(warmup,start,1,prompt.size()+start,true);decode(state,batch,true);}
        check(llama_state_seq_set_data_ext(state.ctx,saved_state.data(),saved_state.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)==saved_state.size(),"checkpoint restore failed");
        check(llama_memory_seq_rm(memory,0,checkpoint,-1),"checkpoint tail trim failed");
        const auto & cells=dynamic_cast<llama_kv_cache_dsv4 *>(memory)->get_raw()->get_swa()->get_cells(0);
        std::ofstream positions(directory+"/restored-cells.json");positions<<"{\"size\":"<<cells.size()<<",\"used_max\":"<<cells.used_max_p1()<<",\"rows\":[";bool first=true;
        for(uint32_t i=0;i<cells.size();++i) if(!cells.is_empty(i)) {positions<<(first?"":",")<<"["<<i<<","<<cells.pos_get(i)<<"]";first=false;}positions<<"]}\n";
        prefill(checkpoint,prompt.size());
    }
    state.measured=true;state.trace=std::fopen((directory+"/raw-trace.jsonl").c_str(),"w");check(state.trace,"trace failed");
    const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream output(directory+"/logits.f32",std::ios::binary),rows(directory+"/rows.jsonl");
    std::vector<uint8_t> eog(vocab);for(int i=0;i<vocab;++i)eog[i]=llama_vocab_is_eog(llama_model_get_vocab(model),i);
    auto record=[&](int index,int row) {
        const float * values=llama_get_logits_ith(state.ctx,row);check(values,"logits missing");
        output.write(reinterpret_cast<const char *>(values),vocab*4);int best=0;
        for(int i=0;i<vocab;++i) {check(std::isfinite(values[i]),"nonfinite logit");if(values[i]>values[best])best=i;}
        int non_eog=-1;for(int i=0;i<vocab;++i)if(!eog[i] && (non_eog<0 || values[i]>values[non_eog]))non_eog=i;
        rows<<"{\"argmax_no_eog\":"<<non_eog<<",\"index\":"<<index<<",\"argmax\":"<<best<<",\"reference\":"<<forced[index]<<"}\n";
    };
    record(0,-1);
    for(int start=0;start<(int)forced.size()-1;) {
        const int n=std::min(width,(int)forced.size()-1-start);fill(forced,start,n,prompt.size()+start,true);decode(state,batch,true);
        for(int col=0;col<n;++col)record(start+col+1,col);start+=n;
    }
    std::fclose(state.trace);output.close();rows.close();
    std::ofstream stats(directory+"/counts.json");stats<<"{\"prompt_tokens\":"<<prompt.size()<<",\"rows\":"<<forced.size()<<",\"vocab\":"<<vocab<<",\"width\":"<<width<<",\"cached\":"<<(cached?"true":"false")<<",\"raw_crops\":"<<state.options.padding_crops<<",\"compressed_crops\":"<<state.options.compressed_padding_crops<<"}\n";
    std::ofstream counts(directory+"/indexer-counts.json");counts<<"{";bool first=true;
    for(const auto & item:state.options.indexer_counts) {counts<<(first?"":",")<<"\""<<item.first<<"\":"<<item.second;first=false;}
    counts<<"}\n";counts.close();
    check(state.options.indexer_counts.empty()==(width==1),"indexer selection count failed");
    llama_batch_free(batch);llama_free(state.ctx);
    std::printf("DONE %s rows=%zu width=%d cached=%d\n",directory.c_str(),forced.size(),width,cached);std::fflush(stdout);
}
int main(int argc,char ** argv) {
    try {
        check(argc==4,"usage: cache-replay MODEL CASES OUTPUT");check(!std::filesystem::exists(argv[3]),"output exists");std::filesystem::create_directories(argv[3]);
        llama_backend_init();ggml_backend_load_all();llama_log_set(logging,nullptr);
        auto mp=llama_model_default_params();std::vector<float> split(llama_max_devices(),0.0f);for(int i=0;i<6;++i)split[i]=i==5?0.4f:1.0f;
        mp.n_gpu_layers=99;mp.split_mode=LLAMA_SPLIT_MODE_LAYER;mp.tensor_split=split.data();mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_AUTO;
        auto * model=llama_model_load_from_file(argv[1],mp);check(model,"model load failed");
        std::ifstream maps("/proc/self/maps");std::ofstream loaded(std::string(argv[3])+"/loaded-libraries.txt");
        for(std::string line;std::getline(maps,line);)if(line.find("/libllama")!=std::string::npos || line.find("/libggml")!=std::string::npos)loaded<<line<<'\n';loaded.close();
        std::ifstream cases(argv[2]);check(bool(cases),"cases open failed");
        for(std::string line;std::getline(cases,line);) {std::istringstream row(line);std::string label,prompt,prefix,warm;int width,cached;check(bool(row>>label>>prompt>>prefix>>width>>cached),"case parse failed");row>>warm;replay(model,tokens(prompt,model),tokens(prefix,model),width,cached,std::string(argv[3])+"/"+label,tokens(warm.empty()?prefix:warm,model));}
        llama_model_free(model);llama_backend_free();return 0;
    } catch(const std::exception & error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
