#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
static void check(bool b,const char *s){if(!b)throw std::runtime_error(s);}
int main(int argc,char **argv){try{
 check(argc==6,"model out gpu fa q8");std::string dir=argv[2];std::filesystem::create_directories(dir);ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=std::stoi(argv[3])?99:0;auto*m=llama_model_load_from_file(argv[1],mp);check(m,"model");
 auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=128;cp.n_ubatch=32;cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=2;cp.flash_attn_type=std::stoi(argv[4])?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;if(std::stoi(argv[5]))cp.type_k=cp.type_v=GGML_TYPE_Q8_0;auto*c=llama_init_from_model(m,cp);check(c,"ctx");int v=llama_vocab_n_tokens(llama_model_get_vocab(m));
 auto decode=[&](int pos,int n,bool output){auto b=llama_batch_init(n,0,1);b.n_tokens=n;for(int i=0;i<n;++i){b.token[i]=(3*(pos+i)+1)%v;b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=output;}check(llama_decode(c,b)==0,"decode");std::vector<float> res;if(output)for(int i=0;i<n;++i){auto*l=llama_get_logits_ith(c,i);res.insert(res.end(),l,l+v);}llama_batch_free(b);return res;};
 for(int n:{65,431,1031}){llama_memory_clear(llama_get_memory(c),true);for(int p=0;p<n;p+=32)decode(p,std::min(32,n-p),false);std::vector<llama_token> tok(n);for(int i=0;i<n;++i)tok[i]=(3*i+1)%v;std::string state=dir+"/state"+std::to_string(n)+".bin";check(llama_state_seq_save_file(c,state.c_str(),0,tok.data(),n)>0,"save");std::vector<float> ref;for(int i=0;i<8;++i){auto a=decode(n+i,1,true);ref.insert(ref.end(),a.begin(),a.end());}llama_memory_clear(llama_get_memory(c),true);size_t nt;std::vector<llama_token> load(n);check(llama_state_seq_load_file(c,state.c_str(),0,load.data(),n,&nt)>0&&nt==n&&load==tok,"load");std::vector<float> after;for(int i=0;i<8;++i){auto a=decode(n+i,1,true);after.insert(after.end(),a.begin(),a.end());}double d=0;int first=-1;for(size_t i=0;i<ref.size();++i){check(std::isfinite(after[i]),"finite");if(ref[i]!=after[i]&&first<0)first=i/v;d=std::max(d,std::fabs(double(ref[i])-after[i]));}for(auto p:{std::pair("fresh",&ref),std::pair("restored",&after)})std::ofstream(dir+"/"+std::to_string(n)+p.first+".f32",std::ios::binary).write((char*)p.second->data(),p.second->size()*4);printf("{\"prefix\":%d,\"exact\":%s,\"first_step\":%d,\"max_abs\":%.12g}\n",n,ref==after?"true":"false",first,d);fflush(stdout);}
 llama_free(c);llama_model_free(m);llama_backend_free();return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
