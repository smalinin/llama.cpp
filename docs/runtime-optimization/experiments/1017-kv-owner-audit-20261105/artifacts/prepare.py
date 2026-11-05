from pathlib import Path
import json,hashlib,subprocess,shutil
R=Path('/home/sergei/Github/llama.cpp');Old=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts';E=R/'docs/runtime-optimization/experiments/1017-kv-owner-audit-20261105';P=E/'artifacts';P.mkdir(parents=True)
(E/'.gitignore').write_text('artifacts/*.so\nartifacts/scopes.jsonl\nartifacts/__pycache__/\nartifacts/scheduler-harness\n!*.log\n')
s=(Old/'probe.cpp').read_text();s=s.replace('#include <atomic>','#include <atomic>\n#include <vector>\n#include <unordered_set>')
a=s.index('static json tensor_info(');b=s.index('\nusing Copy=',a)
s=s[:a]+'''static json tensor_info(const ggml_tensor*t) {
 json j={{"id",uintptr_t(t)},{"name",t->name},{"bytes",ggml_nbytes(t)},{"data",uintptr_t(t->data)},{"buffer",uintptr_t(t->buffer)},{"type",ggml_type_name(t->type)},{"op",ggml_op_name(t->op)},{"flags",t->flags},{"input",bool(t->flags&GGML_TENSOR_FLAG_INPUT)},{"output",bool(t->flags&GGML_TENSOR_FLAG_OUTPUT)},{"view_src",uintptr_t(t->view_src)},{"view_offs",t->view_offs},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"nb",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}}};
 const ggml_tensor*root=t;std::unordered_set<const ggml_tensor*>path;size_t off=0;
 while(root->view_src){if(!path.insert(root).second){j["view_cycle"]=true;break;}off+=root->view_offs;root=root->view_src;}
 j["root"]=uintptr_t(root);j["root_name"]=root->name;j["root_data"]=uintptr_t(root->data);j["root_bytes"]=ggml_nbytes(root);j["root_offset"]=off;j["root_buffer"]=uintptr_t(root->buffer);
 json src=json::array();for(auto*x:t->src)src.push_back(uintptr_t(x));j["src"]=src;
 json op=json::array();for(auto x:t->op_params)op.push_back(x);j["op_params_i32"]=op;
 if(root->buffer){auto*buf=root->buffer;auto*bt=ggml_backend_buffer_get_type(buf);auto*dev=ggml_backend_buft_get_device(bt);j["buffer_name"]=ggml_backend_buffer_name(buf);j["buffer_type"]=ggml_backend_buft_name(bt);j["buffer_owner"]=dev?ggml_backend_dev_name(dev):"unknown";j["buffer_host"]=ggml_backend_buffer_is_host(buf);j["buffer_usage"]=int(ggml_backend_buffer_get_usage(buf));j["buffer_size"]=ggml_backend_buffer_get_size(buf);j["buffer_base"]=uintptr_t(ggml_backend_buffer_get_base(buf));}
 return j;
}
static json graph_description(ggml_cgraph*g){
 json out={{"nodes",json::array()},{"tensors",json::array()}};std::vector<const ggml_tensor*>queue;std::unordered_set<const ggml_tensor*>seen;
 for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);out["nodes"].push_back(uintptr_t(t));queue.push_back(t);}
 for(size_t i=0;i<queue.size();++i){auto*t=queue[i];if(!t||!seen.insert(t).second)continue;out["tensors"].push_back(tensor_info(t));if(t->view_src)queue.push_back(t->view_src);for(auto*x:t->src)if(x)queue.push_back(x);}
 return out;
}
extern "C" const char*dsv_owner_describe_graph(ggml_cgraph*g){static thread_local std::string out;out=graph_description(g).dump();return out.c_str();}
static bool describe_current(){
 if(!current||strcmp(current->kind,"ubatch"))return false;auto*s=current;while(s->parent)s=s->parent;if(strcmp(s->kind,"target_decode"))return false;
 const int pos=current->pos;return pos==0||pos==512||pos==1536||pos==2048||pos==4096||pos==4608||pos==6144||pos==7168||pos==7676;
}
''' + s[b:]
s=s.replace('c.j["success"]=ok;return ok;','c.j["success"]=ok;c.j["scheduler"]=uintptr_t(s);c.j["copies"]=ggml_backend_sched_get_n_copies(s);if(ok&&describe_current())c.j["allocated_graph"]=graph_description(g);return ok;')
s=s.replace('auto rc=fn(b,g);c.j["status"]', 'if(describe_current())c.j["graph_snapshot"]=graph_description(g);auto rc=fn(b,g);c.j["status"]')
s+='''
extern "C" void ggml_backend_event_record(ggml_backend_event_t e,ggml_backend_t b){static auto fn=next_fn<decltype(&ggml_backend_event_record)>("ggml_backend_event_record");if(!current){fn(e,b);return;}Call c("event_record",__builtin_return_address(0),b);c.j["event"]=uintptr_t(e);c.j["event_device"]=e->device?ggml_backend_dev_name(e->device):"null";fn(e,b);}
'''
(P/'probe.cpp').write_text(s);shutil.copy2(Old/'memory_monitor.py',P/'memory_monitor.py')
cmd=['/usr/bin/c++','-O2','-std=c++17','-fPIC','-shared',str(P/'probe.cpp'),*['-I'+str(R/x)for x in ['src','include','common','ggml/src','ggml/include','vendor']],'-ldl','-pthread','-o',str(P/'probe.so')]
with(P/'build.log').open('wb')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=R,check=True)
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
(P/'build.json').write_text(json.dumps({'command':cmd,'source_sha256':sha(P/'probe.cpp'),'probe_sha256':sha(P/'probe.so'),'base_sha256':sha(R/'build-aliceai/bin/libggml-base.so.0'),'runtime_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'scope':'External observer: original ops/events/copies/args/status forwarded, no added GPU events/CUDA API hooks/callbacks/device reads. Public buffer-owner and tensor metadata only; sched_get_tensor_backend deliberately avoided because getter may insert into scheduler hash set. First/last metadata plus complete graph/source/view descriptions on selected target ubatches.'},indent=2)+'\n')
# Adapt the existing scheduler fixture outside tests; describe one real allocated graph.
O=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts/promotion';s=(O/'scheduler.cpp').read_text().replace('#include <algorithm>','#include <algorithm>\n#include <dlfcn.h>');needle='check(ggml_backend_sched_alloc_graph(s,g),"allocate");';assert needle in s
s=s.replace(needle,needle+'''if(variant=="eligible"&&mode==0&&step==0){auto describe=(const char*(*)(ggml_cgraph*))dlsym(RTLD_DEFAULT,"dsv_owner_describe_graph");check(describe,"probe export");auto d=json::parse(describe(g));check(d["nodes"].size()==size_t(ggml_graph_n_nodes(g)),"graph node coverage");bool weight=false,host=false,gpu=false;std::set<uintptr_t>known;for(const auto&t:d["tensors"]){known.insert(t["id"].get<uintptr_t>());check(!t.value("view_cycle",false),"acyclic views");check(t["ne"].size()==4&&t["nb"].size()==4,"layout metadata");if(t["name"]=="identity.weight"){weight=true;check(t["buffer_owner"]=="CUDA0"&&!t["buffer_host"].get<bool>(),"GPU weight owner");}if(t["name"]=="lookup.weight"){host=true;check(t["buffer_host"].get<bool>(),"CPU weight host");}gpu|=t.value("buffer_owner","")=="CUDA1";}check(weight&&host&&gpu,"owner coverage");for(const auto&t:d["tensors"]){check(known.count(t["root"].get<uintptr_t>()),"root covered");for(auto x:t["src"])check(x.get<uintptr_t>()==0||known.count(x.get<uintptr_t>()),"source covered");}std::ofstream(std::getenv("DSV_OWNER_SMOKE_JSON"))<<d.dump(2)<<'\\n';}''').replace('#include <dlfcn.h>','#include <dlfcn.h>\n#include <set>')
(P/'scheduler.cpp').write_text(s);cmd=[x.replace(str(O/'scheduler.cpp'),str(P/'scheduler.cpp')).replace(str(O/'scheduler-harness'),str(P/'scheduler-harness'))for x in json.loads((O/'build.json').read_text())['harness_compile']];cmd+=['-ldl']
with(P/'scheduler-build.log').open('wb')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=R,check=True)
(P/'scheduler-build.json').write_text(json.dumps({'command':cmd},indent=2)+'\n');shutil.copy2('/tmp/prepare-kv-owner-audit.py',P/'prepare.py')
# Same fresh 8K warm/warm/pre/armed/post controller, unchanged production args.
s=(Old/'launch-attribution/run.py').read_text().replace('/1017-prefill-residual-20261105/artifacts/launch-attribution','/1017-kv-owner-audit-20261105/artifacts').replace('19095','19101').replace("='prefill-launch'","='prefill-kv-owner'")
s=s.replace("'kind':'", "'kind':'KV root/buffer/source/event ownership audit, no CUDA API interposition or runtime change. ")
(P/'run.py').write_text(s)
print('Owner observer and external scheduler fixture built.',flush=True)
