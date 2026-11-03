#include "llama-model.h"
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <dlfcn.h>
#include <fstream>
#include <set>
#include <cstdlib>
#include <cstdio>
extern "C" llama_model * llama_model_load_from_file(const char * path,llama_model_params p){
 static auto next=reinterpret_cast<decltype(&llama_model_load_from_file)>(dlsym(RTLD_NEXT,"llama_model_load_from_file"));if(!next){std::abort();}auto*m=next(path,p);if(!m||p.no_alloc)return m;const char*dest=std::getenv("DSV_PLACEMENT_FILE");if(!dest)return m;
 using J=nlohmann::ordered_json;char arch[64]={};llama_model_meta_val_str(m,"general.architecture",arch,sizeof arch);J j={{"arch",arch},{"path",path},{"layers",J::array()}};size_t gpu=0,host=0;int full=0,total=0;
 for(size_t il=0;il<m->layers.size();++il){const auto&l=m->layers[il];J row={{"layer",il},{"device",ggml_backend_dev_name(m->dev_layer(il))},{"tensors",J::array()}};std::set<ggml_tensor*>seen;bool allgpu=true;int nex=0;
  for(auto*t:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_gate_up_exps,l.ffn_down_exps})if(t){auto*b=t->view_src?t->view_src:t;if(!seen.insert(b).second)continue;bool h=ggml_backend_buffer_is_host(b->buffer);size_t bytes=ggml_nbytes(b);(h?host:gpu)+=bytes;allgpu &= !h;nex=b->ne[2];row["tensors"].push_back({{"name",b->name},{"bytes",bytes},{"buffer",ggml_backend_buffer_name(b->buffer)},{"host",h},{"type",ggml_type_name(b->type)}});}
  if(!seen.empty()){j["layers"].push_back(row);total+=nex;if(allgpu)full+=nex;}}
 j["routed_gpu_bytes"]=gpu;j["routed_host_bytes"]=host;j["total_expert_bundles"]=total;j["gpu_complete_bundles"]=full;std::ofstream f(dest,std::ios::app);f<<j.dump()<<'\n';return m;
}
