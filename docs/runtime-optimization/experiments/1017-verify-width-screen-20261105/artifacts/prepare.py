from pathlib import Path
import json,shutil,subprocess,hashlib
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-verify-width-screen-20261105';P=E/'artifacts';O=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';assert not E.exists();P.mkdir(parents=True)
(E/'.gitignore').write_text('artifacts/*.so\nartifacts/*-harness\nartifacts/__pycache__/\n!*.log\n')
shutil.copy2(O/'memory_monitor.py',P/'memory_monitor.py');cfg=json.loads((O/'native-results.json').read_text())['config'];(P/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');B=R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105/artifacts';(P/'inputs.json').write_text((B/'inputs.json').read_text())
header='''#pragma once
#include <cstdint>
static const char * counter_names[]={"graphs","nodes","float_matmul","float_single_row_matmul","moe_matmul","get_rows","backend_sync","scheduler_sync","event_sync","event_wait","copy_async_api","copy_sync_api","cuda_graph_launch","cuda_graph_instantiate","cuda_kernel_launch","prepare_inputs","allocate_graph","quant_matmul"};
constexpr int counter_count=sizeof(counter_names)/sizeof(counter_names[0]);
''';(P/'counts.h').write_text(header)
probe='''#include "counts.h"
#include "ggml-backend.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
static std::atomic<bool> active{false};
static std::atomic<uint64_t> counts[counter_count];
template<class F> static F next(const char *name){auto *p=dlsym(RTLD_NEXT,name);if(!p){fprintf(stderr,"COUNTER missing %s\\n",name);abort();}return reinterpret_cast<F>(p);}
static void add(int i,uint64_t n=1){if(active.load(std::memory_order_relaxed))counts[i].fetch_add(n,std::memory_order_relaxed);}
extern "C" void dsv_width_count_begin(){active.store(false);for(auto &x:counts)x.store(0);active.store(true);}
extern "C" void dsv_width_count_end(uint64_t *out){active.store(false);for(int i=0;i<counter_count;++i)out[i]=counts[i].load();}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,ggml_cgraph *g){static auto f=next<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");if(active.load(std::memory_order_relaxed)){add(0);int n=ggml_graph_n_nodes(g);add(1,n);for(int i=0;i<n;++i){auto *t=ggml_graph_node(g,i);if(t->op==GGML_OP_MUL_MAT){bool q=ggml_is_quantized(t->src[0]->type);add(q?17:2);if(!q&&t->src[1]->ne[1]==1)add(3);}else if(t->op==GGML_OP_MUL_MAT_ID)add(4);else if(t->op==GGML_OP_GET_ROWS)add(5);}}return f(b,g);}
extern "C" void ggml_backend_synchronize(ggml_backend_t b){static auto f=next<decltype(&ggml_backend_synchronize)>("ggml_backend_synchronize");add(6);f(b);}
extern "C" void ggml_backend_sched_synchronize(ggml_backend_sched_t s){static auto f=next<decltype(&ggml_backend_sched_synchronize)>("ggml_backend_sched_synchronize");add(7);f(s);}
extern "C" void ggml_backend_event_synchronize(ggml_backend_event_t e){static auto f=next<decltype(&ggml_backend_event_synchronize)>("ggml_backend_event_synchronize");add(8);f(e);}
extern "C" void ggml_backend_event_wait(ggml_backend_t b,ggml_backend_event_t e){static auto f=next<decltype(&ggml_backend_event_wait)>("ggml_backend_event_wait");add(9);f(b,e);}
extern "C" void ggml_backend_tensor_copy_async(ggml_backend_t a,ggml_backend_t b,const ggml_tensor *s,ggml_tensor *d){static auto f=next<decltype(&ggml_backend_tensor_copy_async)>("ggml_backend_tensor_copy_async");add(10);f(a,b,s,d);}
extern "C" void ggml_backend_tensor_copy(const ggml_tensor *s,ggml_tensor *d){static auto f=next<decltype(&ggml_backend_tensor_copy)>("ggml_backend_tensor_copy");add(11);f(s,d);}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g,cudaStream_t s){static auto f=next<decltype(&cudaGraphLaunch)>("cudaGraphLaunch");add(12);return f(g,s);}
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t *e,cudaGraph_t g,unsigned long long flags){static auto f=next<decltype(&cudaGraphInstantiate)>("cudaGraphInstantiate");add(13);return f(e,g,flags);}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *c,const void *f,void **a){static auto fn=next<decltype(&cudaLaunchKernelExC)>("cudaLaunchKernelExC");add(14);return fn(c,f,a);}
extern "C" void ggml_backend_sched_prepare_inputs(ggml_backend_sched_t s){static auto f=next<decltype(&ggml_backend_sched_prepare_inputs)>("ggml_backend_sched_prepare_inputs");add(15);f(s);}
extern "C" bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t s,ggml_cgraph *g){static auto f=next<decltype(&ggml_backend_sched_alloc_graph)>("ggml_backend_sched_alloc_graph");add(16);return f(s,g);}
''';(P/'counts.cpp').write_text(probe)
old=(O/'precision.cpp').read_text();helpers=old[:old.index('static json draft(')];start=old[old.index('int main('):old.index('for(auto&corpus:input[')];start=start.replace('p.speculative.draft.p_min=0;','').replace('std::cerr<<"READY\\n";','need(llama_n_rs_seq(t)==5,"five rollback snapshots");need(t->get_cparams().pipeline_parallel,"pipeline enabled");out["n_rs_seq"]=llama_n_rs_seq(t);std::cerr<<"READY\\n";');start=start.replace('{"faults",json::array()}','{"protocol","Fixed-prefix target all-logits calls, restore/sync outside timer. No proposal selection or measured feature bridge. Raw-logit output transport differs from server backend sampling; not native end-to-end TG."}')
main='''
using Begin=void(*)();using End=void(*)(uint64_t*);auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"dsv_width_count_begin"));auto end=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"dsv_width_count_end"));need(bool(begin)==bool(end),"counter pair");out["instrumented"]=bool(begin);auto tok=input["tokens"].get<Tokens>();need(tok.size()>90118,"input length");
for(int prefix:{8192,90112}){
 llama_memory_clear(llama_get_memory(t),true);llama_memory_clear(llama_get_memory(d),true);common_speculative_set_state(spec.get(),0,{});
 for(int pos=0;pos<prefix;){int n=std::min(8192,prefix-pos);apply(t,d,spec.get(),part(tok,pos,n),pos,false,false);pos+=n;std::cerr<<"PREFILL "<<prefix<<' '<<pos<<'\\n';}
 auto base=state(t);auto basehash=sha(base);std::map<int,std::string> ref,states;Floats one;json cases=json::array();
 for(int pass=0;pass<2;++pass)for(int i=0;i<6;++i){int n=pass?6-i:i+1;auto b=llama_batch_init(n,0,1);b.n_tokens=n;for(int k=0;k<n;++k){b.token[k]=tok[prefix+k];b.pos[k]=prefix+k;b.n_seq_id[k]=1;b.seq_id[k][0]=0;b.logits[k]=true;}
  int warm=pass?1:2;
  for(int rep=-warm;rep<5;++rep){restore(t,base);llama_synchronize(t);uint64_t counts[counter_count]{};auto perf=llama_perf_context(t);if(begin)begin();auto before=std::chrono::steady_clock::now();int rc=llama_decode(t,b);llama_synchronize(t);auto after=std::chrono::steady_clock::now();if(end)end(counts);need(rc==0,"verification decode");auto v=logits(t,n);auto h=sha(v);if(!ref.count(n))ref[n]=h;else need(ref[n]==h,"same-width repeat logits");if(n==1&&one.empty())one=v;float delta=0;for(int k=0;k<nv;++k)delta=std::max(delta,std::fabs(v[k]-one[k]));json row={{"pass",pass},{"queries",n},{"rep",rep},{"kind",rep<0?"warmup":"measured"},{"us",std::chrono::duration<double,std::micro>(after-before).count()},{"logits_sha256",h},{"first_row_max_abs_vs_width1",delta},{"first_row_argmax",int(std::max_element(v.begin(),v.begin()+nv)-v.begin())},{"splits",ggml_backend_sched_get_n_splits(t->get_sched())},{"graphs_reused_delta",llama_perf_context(t).n_reused-perf.n_reused}};if(begin){json c;for(int k=0;k<counter_count;++k)c[counter_names[k]]=counts[k];row["counts"]=c;}
   if(rep==4){auto st=sha(state(t));row["target_state_sha256"]=st;if(!states.count(n))states[n]=st;else need(states[n]==st,"same-width state replay");}
   cases.push_back(row);out["progress"]={{"prefix",prefix},{"pass",pass},{"queries",n},{"rep",rep}};save();
  }llama_batch_free(b);std::cerr<<"WIDTH "<<prefix<<' '<<pass<<' '<<n<<'\\n';
 }
 restore(t,base);need(sha(state(t))==basehash,"base restore exact");out["cases"].push_back({{"prefix",prefix},{"base_state_sha256",basehash},{"base_state_bytes",base.size()},{"fixed_proposal_ids",part(tok,prefix,6)},{"rows",cases},{"same_width_logits_state_exact",true},{"base_restore_exact",true}});save();
}
out["complete"]=true;save();std::cerr<<"WIDTH_SCREEN_COMPLETE\\n";
}catch(const std::exception&e){out["error"]=e.what();save();std::cerr<<"FAIL "<<e.what()<<'\\n';return 1;}return 0;}
'''
source=helpers+'#include "counts.h"\n#include <chrono>\n#include <algorithm>\n#include <dlfcn.h>\n'+start+main;(P/'width.cpp').write_text(source)
cmd=json.loads((O/'precision-build.json').read_text())['command'];cmd=[x.replace(str(O/'precision.cpp'),str(P/'width.cpp')).replace(str(O/'precision-harness'),str(P/'width-harness'))for x in cmd]
probe_cmd=['/usr/bin/c++','-O2','-std=c++17','-shared','-fPIC','-I'+str(R/'ggml/include'),'-I/usr/local/cuda/include',str(P/'counts.cpp'),'-ldl','-o',str(P/'counts.so')]
with(P/'build.log').open('wb')as log:
 for c in [cmd,probe_cmd]:subprocess.run(c,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest();(P/'build.json').write_text(json.dumps({'complete':True,'commands':[cmd,probe_cmd],'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'harness_sha256':sha(P/'width-harness'),'probe_sha256':sha(P/'counts.so'),'source_sha256':sha(P/'width.cpp'),'probe_source_sha256':sha(P/'counts.cpp'),'private_library':str(O/'candidate/libllama.so.0'),'private_library_sha256':sha(O/'candidate/libllama.so.0'),'note':'Adapt existing precision harness load/prefill/restore/logits helpers. External diagnostic only, no runtime changes; eval callbacks/CUPTI/backtrace absent. Counters are separate process, without per-call timestamps or JSON.'},indent=2)+'\n')
(E/'REPORT.md').write_text('# Verification widths 1..6 on the new host\n\nStatus: read-only diagnostic prepared; no runtime/default change. Fixed-prefix raw target all-logits calls at8K/88K, with the same six proposed token IDs truncated per width and target snapshots restored outside synchronized timers. Two passes in reverse order,10measured+3warmups per width/prefix. Separate native and lightweight-counter processes; costs include raw-logit transfer, not native server backend sampling or total TG. Full model and draft/feature registration remain loaded; bridge itself is outside the measured calls. Final comparisons require exact per-width repeat outputs/state and matching instrumented/uninstrumented results, actual>=512MiB free, clean exit and untouched modules.\n')
shutil.copy2('/tmp/prepare-verify-width-screen.py',P/'prepare.py');print('BUILT',E)
