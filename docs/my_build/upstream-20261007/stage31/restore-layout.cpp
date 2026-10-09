#include "llama.h"
#include "llama-memory-hybrid.h"
#include "llama-batch.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool b,const char * s){if(!b)throw std::runtime_error(s);}
static void occupy(llama_kv_cache * kv,int count){
 if(!count)return;llama_batch_allocr allocator(1);auto b=allocator.ubatch_reserve(count,1);b.seq_id_unq[0]=1;
 for(int i=0;i<count;++i){b.token[i]=1;b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i]=b.seq_id_unq;}
 auto slot=kv->find_slot(b,false);check(!slot.empty(),"dummy allocation");kv->apply_ubatch(slot,b);
}
int main(int argc,char ** argv){
 try{
 check(argc==5,"model output gpu q8");std::string dir=argv[2];std::filesystem::create_directories(dir);
 ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=std::stoi(argv[3])?99:0;auto * model=llama_model_load_from_file(argv[1],mp);check(model,"model");
 auto cp=llama_context_default_params();cp.n_ctx=65536;cp.n_batch=512;cp.n_ubatch=64;cp.n_seq_max=3;cp.kv_unified=true;cp.n_threads=2;cp.n_threads_batch=2;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;if(std::stoi(argv[4]))cp.type_k=cp.type_v=GGML_TYPE_Q8_0;
 auto * ctx=llama_init_from_model(model,cp);check(ctx,"context");auto * mem=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));check(mem,"memory");int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));std::vector<llama_token> tokens;
 auto decode=[&](int pos,int count,bool output){auto b=llama_batch_init(count,0,1);b.n_tokens=count;for(int i=0;i<count;++i){b.token[i]=(3*(pos+i)+1)%vocab;b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=output;if(!output)tokens.push_back(b.token[i]);}check(llama_decode(ctx,b)==0,"decode");std::vector<float> result;if(output){for(int i=0;i<count;++i){auto * l=llama_get_logits_ith(ctx,i);result.insert(result.end(),l,l+vocab);}}llama_batch_free(b);return result;};
 for(int pos=0;pos<431;pos+=64)decode(pos,std::min(64,431-pos),false);
 std::string state=dir+"/prefix.bin";check(llama_state_seq_save_file(ctx,state.c_str(),0,tokens.data(),tokens.size())>0,"save file");auto reference=decode(431,4,true);
 for(int offset:{0,499,4000,33000,65000}){
  llama_memory_clear(mem,true);occupy(mem->get_mem_attn(),offset);occupy(mem->get_mem_idx(),offset);
  std::vector<llama_token> restored(tokens.size());size_t count=0;check(llama_state_seq_load_file(ctx,state.c_str(),0,restored.data(),restored.size(),&count)>0,"load file");check(count==tokens.size()&&restored==tokens,"tokens");
  check(llama_memory_seq_rm(mem,1,0,-1),"remove dummy");const auto & cells=mem->get_mem_attn()->get_cells(0);int first=-1,last=-1;for(uint32_t j=0;j<cells.size();++j){if(cells.seq_has(j,0)){if(first<0)first=j;last=j;}}
  auto value=decode(431,4,true);double difference=0;for(size_t i=0;i<value.size();++i){check(std::isfinite(value[i]),"nonfinite");difference=std::max(difference,std::fabs(double(value[i])-reference[i]));}
  std::ofstream(dir+"/offset"+std::to_string(offset)+".f32",std::ios::binary).write((char*)value.data(),value.size()*sizeof(float));
  std::printf("{\"offset\":%d,\"first_cell\":%d,\"last_cell\":%d,\"logits_equal\":%s,\"max_abs\":%.12g}\n",offset,first,last,std::memcmp(value.data(),reference.data(),value.size()*sizeof(float))==0?"true":"false",difference);std::fflush(stdout);
 }
 std::ofstream(dir+"/fresh.f32",std::ios::binary).write((char*)reference.data(),reference.size()*sizeof(float));llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
 }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
