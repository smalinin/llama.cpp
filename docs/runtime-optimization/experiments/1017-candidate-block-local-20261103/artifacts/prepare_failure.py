from pathlib import Path
P=Path(__file__).parent;s=(P/'operator.cpp').read_text().replace('#include "ggml-cpu.h"','#include "ggml-cpu.h"\n#include "ggml-backend-impl.h"')
s=s.replace('using json=nlohmann::ordered_json;','''using json=nlohmann::ordered_json;
static decltype(ggml_backend_i::graph_compute) original_compute=nullptr;
static bool armed=false;static int injected=0;
static ggml_status fail_once(ggml_backend_t b,ggml_cgraph*g){if(armed){armed=false;++injected;return GGML_STATUS_FAILED;}return original_compute(b,g);}''')
s=s.replace('bool local,int iters=2)', 'bool local,int iters=2,bool fault=false)')
s=s.replace('for(auto*p:b)need(p,"backend");','for(auto*p:b)need(p,"backend");original_compute=b[1]->iface.graph_compute;if(fault)b[1]->iface.graph_compute=fail_once;')
s=s.replace('need(ggml_backend_sched_graph_compute(s,g)==GGML_STATUS_SUCCESS,"compute");','''if(fault&&iter==1)armed=true;auto status=ggml_backend_sched_graph_compute(s,g);if(fault&&iter==1){need(status==GGML_STATUS_FAILED&&!armed,"fault missing");status=ggml_backend_sched_graph_compute(s,g);}need(status==GGML_STATUS_SUCCESS,"compute/retry");''')
s=s.replace('ggml_backend_sched_free(s);','if(fault)b[1]->iface.graph_compute=original_compute;ggml_backend_sched_free(s);')
s=s[:s.index('int main(')]+'''int main(int argc,char**argv){json j={{"complete",false},{"scope","External transport-operator failure control. A test-only wrapper makes CUDA1 graph submission fail once after earlier split submission; existing synchronous scheduler wrapper drains before retry. No production iface mutation. Tests buffer reuse/result recovery, not full model KV abort semantics."},{"cases",json::array()}};try{ggml_backend_load_all();for(bool local:{false,true}){int before=injected;auto a=run(16391,7,0,local,3,false),b=run(16391,7,0,local,3,true);need(a.masks==b.masks&&a.tops==b.tops&&a.blocks==b.blocks,"retry mismatch");need(injected==before+1,"injection count");j["cases"].push_back({{"local",local},{"exact",true},{"injected",1}});}j["complete"]=true;}catch(const std::exception&e){j["error"]=e.what();}std::ofstream(argv[1])<<j.dump(2)<<'\\n';return j["complete"]?0:1;}
''';(P/'operator-failure.cpp').write_text(s)
