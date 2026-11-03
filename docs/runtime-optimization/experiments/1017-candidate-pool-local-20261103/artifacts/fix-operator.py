from pathlib import Path
import json,hashlib,subprocess,os
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp')
# Keep the invalid-route run as evidence. It must not count as a two-GPU gate.
s=(P/'operator.cpp').read_text();assert "need(ggml_backend_sched_alloc_graph"not in s
(P/'operator-initial-manifest.json').write_text(json.dumps({'accepted':False,'reason':'Implicit graph_compute reset manual backend hints; CUDA1 compute_calls0. Numerical match did not exercise two routes.','source_sha256':hashlib.file_digest((P/'operator.cpp').open('rb'),'sha256').hexdigest(),'binary_sha256':hashlib.file_digest((P/'operator').open('rb'),'sha256').hexdigest()},indent=2)+'\n')
s=s.replace('Outputs z;','need(ggml_backend_sched_alloc_graph(s,g),"explicit allocate");need(ggml_backend_sched_get_tensor_backend(s,out)==b[1],"selector placement");if(pool)need(ggml_backend_sched_get_tensor_backend(s,pool)==b[local?0:1],"pool placement");Outputs z;')
a=s.index('auto a=run(b,16391,511,0,false,32)');b=s.index('j["complete"]=true;',a)
s=s[:a]+'''for(auto*x:b)ggml_backend_free(x);Outputs fixed[2];for(int mode=0;mode<2;++mode){std::cerr<<"FIXED_BEGIN "<<mode<<'\\n';ggml_backend_t bb[]={ggml_backend_cuda_init(0),ggml_backend_cuda_init(1),ggml_backend_cpu_init()};fixed[mode]=run(bb,16391,511,0,mode==1,32);for(auto*x:bb)ggml_backend_free(x);std::cerr<<"FIXED_END "<<mode<<'\\n';}need(fixed[0].mask==fixed[1].mask&&fixed[0].top==fixed[1].top&&fixed[0].pool==fixed[1].pool,"repeat32 mismatch");j["repeat32"]={{"rows",16391},{"queries",511},{"iterations",32},{"exact",true}};j["actual_placement_checked"]=true;'''+s[b:]
s=s.replace('Existing candidate operators, same selector CUDA1, pool/preparation CUDA1 vs CUDA0.', 'Existing candidate operators, explicit graph allocation preserves checked backend hints, selector CUDA1, pool/preparation CUDA1 vs CUDA0.')
(P/'operator-final.cpp').write_text(s)
with(P/'compile-operator-final.log').open('wb')as f:subprocess.run(['g++','-std=c++17','-O2',str(P/'operator-final.cpp'),'-Iggml/include','-Ivendor','-Lbuild-aliceai/bin','-Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin','-lggml','-lggml-base','-lggml-cpu','-lggml-cuda','-o',str(P/'operator-final')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
e={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};e.update(CUDA_DEVICE_ORDER='FASTEST_FIRST',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',LD_LIBRARY_PATH=str(R/'build-aliceai/bin'),GGML_CUDA_ARGSORT_RADIX='1')
with(P/'operator-final.log').open('wb')as f:subprocess.run([str(P/'operator-final'),str(P/'operator-final-results.json')],cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
x=json.load(open(P/'operator-final-results.json'));assert x['complete']and x['actual_placement_checked']and len(x['cases'])==80 and x['repeat32']['exact'];print('OPERATOR_FINAL_DONE',flush=True)
