#include "llama-context.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "speculative.h"
#include <nvtx3/nvToolsExt.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" bool dsv_profile_active();
extern "C" int dsv_profile_phase();
static thread_local const char * role="other";
template<class F>static F next(const char*n){auto*p=dlsym(RTLD_NEXT,n);if(!p)abort();return reinterpret_cast<F>(p);}
struct Range{bool on;Range(const char*n):on(dsv_profile_active()){if(on)nvtxRangePushA(n);}~Range(){if(on)nvtxRangePop();}};
int llama_context::decode(const llama_batch&b){using F=int(*)(llama_context*,const llama_batch&);static auto fn=next<F>("_ZN13llama_context6decodeERK11llama_batch");char arch[64]{};llama_model_meta_val_str(llama_get_model(this),"general.architecture",arch,sizeof arch);auto old=role;role=!strcmp(arch,"deepseek41")?"target":"draft";char label[160];snprintf(label,sizeof label,"decode.%s phase=%d pos=%d tokens=%d",role,dsv_profile_phase(),b.pos?b.pos[0]:-1,b.n_tokens);Range r(label);auto ret=fn(this,b);role=old;return ret;}
llm_graph_result*llama_context::process_ubatch(const llama_ubatch&b,llm_graph_type gt,llama_memory_context_i*m,ggml_status&s,int kv){using F=llm_graph_result*(*)(llama_context*,const llama_ubatch&,llm_graph_type,llama_memory_context_i*,ggml_status&,int);static auto fn=next<F>("_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_statusi");char label[160];snprintf(label,sizeof label,"ubatch.%s phase=%d pos=%d tokens=%u kv_only=%d",role,dsv_profile_phase(),b.pos?b.pos[0]:-1,b.n_tokens,kv);Range r(label);return fn(this,b,gt,m,s,kv);}
bool common_speculative_process(common_speculative*s,const llama_batch&b){using F=bool(*)(common_speculative*,const llama_batch&);static auto fn=next<F>("_Z26common_speculative_processP18common_speculativeRK11llama_batch");Range r(dsv_profile_phase()==1?"bridge.prefill":"bridge.generation");return fn(s,b);}
void common_speculative_draft(common_speculative*s){using F=void(*)(common_speculative*);static auto fn=next<F>("_Z24common_speculative_draftP18common_speculative");Range r("proposal.generate");fn(s);}
