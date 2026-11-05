#include "arg.h"
#include "common.h"
#include "fit.h"
#include "speculative.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <chrono>
using json=nlohmann::ordered_json;
void source_memory_breakdown_print(const llama_context *);
#define common_fit_params source_fit_params_copy
#define common_get_device_memory_data source_get_memory_data_copy
#define common_fit_print source_fit_print_copy
#define common_memory_breakdown_print source_memory_breakdown_print
#include "/home/sergei/Github/llama.cpp/common/fit.cpp"
#undef common_fit_params
#undef common_get_device_memory_data
#undef common_fit_print
#undef common_memory_breakdown_print
static void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
int main(int argc,char**argv){json j={{"complete",false},{"metadata_only",true}};std::filesystem::path output;auto save=[&](){if(!output.empty()){std::ofstream f(output);f<<j.dump(2)<<'\n';}};try{
 require(argc>4,"OUTPUT measure|fit CLI");output=argv[1];bool fit=std::string(argv[2])=="fit";std::vector<char*>args{argv[0]};for(int i=3;i<argc;++i)args.push_back(argv[i]);common_init();common_params p;require(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"parse");llama_backend_init();
 const auto limits=common_speculative_get_output_limits(p.n_batch,p.n_parallel,common_speculative_n_max(&p.speculative));p.n_outputs_max=std::max(1,limits.total);p.n_outputs_max_per_seq=std::max(1,limits.per_seq);
 auto mp=common_model_params_to_llama(p);mp.load_mtp=false;auto cp=common_context_params_to_llama(p);auto pd=common_base_params_to_speculative(p);auto md=common_model_params_to_llama(pd);md.tensor_split=p.tensor_split;auto cd=common_context_params_to_llama(pd);cd.n_rs_seq=0;
 common_fit_extra_model extra{pd.model.path.c_str(),&md,&cd,false,false,true,p.speculative.draft.backend_sampling};j["requested_context"]=cp.n_ctx;j["action"]=fit?"fit":"measure";auto begin=std::chrono::steady_clock::now();
 if(fit){j["fit_status"]=int(common_fit_params(p.model.path.c_str(),&mp,&cp,p.tensor_split,p.tensor_buft_overrides.data(),p.fit_params_target.data(),p.fit_params_min_ctx,&extra,GGML_LOG_LEVEL_WARN));save();if(j["fit_status"]!=0){j["complete"]=true;save();return 0;}}
 cd.n_ctx=cp.n_ctx;std::vector<ggml_backend_dev_t>td,dd;uint32_t ngl=0,nct=0,nex=0;auto mainmem=common_get_device_memory_data(p.model.path.c_str(),&mp,&cp,td,ngl,nct,nex,GGML_LOG_LEVEL_WARN);auto draftmem=common_get_device_memory_data_impl(pd.model.path.c_str(),&md,&cd,dd,ngl,nct,nex,GGML_LOG_LEVEL_WARN,false,p.model.path.c_str(),&mp,&cp,true,p.speculative.draft.backend_sampling);
 j["memory"]=json::array();bool fits=true;for(size_t i=0;i<td.size();++i){llama_memory_breakdown_data peer;for(size_t k=0;k<dd.size();++k)if(td[i]==dd[k]){peer.model+=draftmem[k].mb.model;peer.context+=draftmem[k].mb.context;peer.compute+=draftmem[k].mb.compute;}int64_t used=mainmem[i].model+mainmem[i].context+mainmem[i].compute+peer.total();int64_t left=mainmem[i].free-used;fits &= left>=512ll*1048576;j["memory"].push_back({{"device",ggml_backend_dev_name(td[i])},{"free_before",mainmem[i].free},{"total",mainmem[i].total},{"main_model",mainmem[i].model},{"main_context",mainmem[i].context},{"main_compute",mainmem[i].compute},{"draft_and_taps_model",peer.model},{"draft_and_taps_context",peer.context},{"draft_and_taps_compute",peer.compute},{"projected_free",left}});}
 j["fits_512mib"]=fits;j["context"]=cp.n_ctx;j["gpu_layers"]=mp.n_gpu_layers;j["target_rs"]=cp.n_rs_seq;j["target_outputs_per_seq"]=cp.n_outputs_max_per_seq;j["split"]=std::vector<float>(p.tensor_split,p.tensor_split+td.size());j["overrides"]=json::array();if(mp.tensor_buft_overrides)for(auto*x=mp.tensor_buft_overrides;x->pattern;++x)j["overrides"].push_back({{"pattern",x->pattern},{"buffer",ggml_backend_buft_name(x->buft)}});
 auto noalloc=mp;noalloc.no_alloc=true;noalloc.load_mode=LLAMA_LOAD_MODE_NONE;auto*m=llama_model_load_from_file(p.model.path.c_str(),noalloc);require(m,"layout model");j["shared_routes"]=json::array();j["routes"]=json::array();size_t gpu=0,host=0;int totalbundles=0,gpubundles=0;
 for(size_t il=0;il<m->layers.size();++il){const auto&l=m->layers[il];if(il==6||il==29)for(auto*t:{l.ffn_gate_shexp,l.ffn_up_shexp,l.ffn_down_shexp})if(t){auto*b=t->view_src?t->view_src:t;j["shared_routes"].push_back({{"name",b->name},{"type",ggml_type_name(b->type)},{"bytes",ggml_nbytes(b)},{"buffer",ggml_backend_buffer_name(b->buffer)}});}json row={{"layer",il},{"device",ggml_backend_dev_name(m->dev_layer(il))},{"tensors",json::array()}};std::set<ggml_tensor*>seen;bool allgpu=true;int experts=0;for(auto*t:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_gate_up_exps,l.ffn_down_exps})if(t){auto*base=t->view_src?t->view_src:t;if(!seen.insert(base).second)continue;bool h=ggml_backend_buffer_is_host(base->buffer);auto nb=ggml_nbytes(base);(h?host:gpu)+=nb;allgpu &= !h;experts=base->ne[2];row["tensors"].push_back({{"name",base->name},{"type",ggml_type_name(base->type)},{"bytes",nb},{"buffer",ggml_backend_buffer_name(base->buffer)},{"host",h}});}if(!seen.empty()){totalbundles+=experts;if(allgpu)gpubundles+=experts;j["routes"].push_back(row);}}
 j["routed_gpu_bytes"]=gpu;j["routed_host_bytes"]=host;j["total_expert_bundles"]=totalbundles;j["gpu_complete_bundles"]=gpubundles;llama_model_free(m);j["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();j["complete"]=true;save();return 0;
 }catch(const std::exception&e){j["error"]=e.what();save();std::cerr<<e.what()<<'\n';return 1;}}
