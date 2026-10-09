#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
static void check(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static std::vector<llama_token> read(const char*name){std::ifstream f(name,std::ios::binary|std::ios::ate);check(bool(f),"tokens open");size_t n=f.tellg();check(n%4==0,"tokens size");std::vector<llama_token> v(n/4);f.seekg(0);check(bool(f.read((char*)v.data(),n)),"tokens read");return v;}
int main(int argc,char**argv){try{
 check(argc==8,"MODEL GPU FA Q8 PROMPT TOKENS OUT");ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();int gpu=std::stoi(argv[2]);mp.n_gpu_layers=gpu?99:0;
 std::vector<float> split(llama_max_devices(),0);for(int i=0;i<6;++i)split[i]=i==5?.4f:1.f;if(gpu==6){mp.tensor_split=split.data();mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_AUTO;}
 auto*m=llama_model_load_from_file(argv[1],mp);check(m,"model");int vocab=llama_vocab_n_tokens(llama_model_get_vocab(m));std::vector<llama_token> prompt,forced;
 if(std::string(argv[5])!="-"){prompt=read(argv[5]);forced=read(argv[6]);}else{prompt.resize(509);forced.resize(272);for(size_t i=0;i<prompt.size();++i)prompt[i]=(3*i+1)%vocab;for(size_t i=0;i<forced.size();++i)forced[i]=(3*(i+prompt.size())+1)%vocab;}
 std::ofstream summary(argv[7]);std::vector<float> reference;
 for(int mode=0;mode<5;++mode){auto cp=llama_context_default_params();cp.n_ctx=8192;cp.n_batch=2048;cp.n_ubatch=512;cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=gpu==6?12:2;cp.n_rs_seq=std::min(mode,3);cp.n_outputs_max=cp.n_outputs_max_per_seq=4;cp.kv_unified=false;cp.swa_full=false;cp.type_k=cp.type_v=std::stoi(argv[4])?GGML_TYPE_Q8_0:GGML_TYPE_F16;cp.flash_attn_type=std::stoi(argv[3])?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto*c=llama_init_from_model(m,cp);check(c,"ctx");auto mem=llama_get_memory(c);llama_memory_clear(mem,true);auto batch=llama_batch_init(2048,0,1);
 auto decode=[&](const std::vector<llama_token>&tokens,int start,int n,int pos,bool outputs){batch.n_tokens=n;for(int i=0;i<n;++i){batch.token[i]=tokens[start+i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=outputs || i+1==n;}check(llama_decode(c,batch)==0,"decode");llama_synchronize(c);};
 for(size_t p=0;p<prompt.size();p+=2048)decode(prompt,p,std::min<size_t>(2048,prompt.size()-p),p,false);
 int differing=0,first=-1,greedy=0,compared=0;double maxabs=0;
 for(int p=0;p<(int)forced.size();){int n=std::min(1+std::min(mode,3),(int)forced.size()-p);check(llama_memory_seq_rm(mem,0,prompt.size()+p,-1),"rollback");decode(forced,p,n,prompt.size()+p,true);int advance=mode==4?std::min(n,1+(p%3)):n;
 for(int i=0;i<advance;++i){auto*l=llama_get_logits_ith(c,i);check(l,"logits");if(mode==0){reference.insert(reference.end(),l,l+vocab);}else{auto*r=reference.data()+(p+i)*vocab;bool exact=memcmp(l,r,vocab*4)==0;if(!exact){++differing;if(first<0)first=p+i;}int arg=0,refarg=0;for(int k=0;k<vocab;++k){check(std::isfinite(l[k]),"finite");maxabs=std::max(maxabs,std::abs(double(l[k])-r[k]));if(l[k]>l[arg])arg=k;if(r[k]>r[refarg])refarg=k;}greedy+=arg!=refarg;}++compared;}
 p+=advance;}
 summary<<"{\"mode\":"<<mode<<",\"rows\":"<<compared<<",\"differing\":"<<differing<<",\"first\":"<<first<<",\"greedy_differing\":"<<greedy<<",\"max_abs\":"<<maxabs<<"}\n";summary.flush();printf("mode %d differing %d first %d greedy %d max %.9g\n",mode,differing,first,greedy,maxabs);fflush(stdout);llama_batch_free(batch);llama_free(c);}
 llama_model_free(m);llama_backend_free();return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
