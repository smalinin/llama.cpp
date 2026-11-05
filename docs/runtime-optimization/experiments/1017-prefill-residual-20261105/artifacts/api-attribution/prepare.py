from pathlib import Path
import subprocess,json,hashlib,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts';D=P/'api-attribution';D.mkdir(exist_ok=True);shutil.copy2(P/'memory_monitor.py',D/'memory_monitor.py')
s=(P/'probe.cpp').read_text();s=s.replace('#include <atomic>','#include <atomic>\n#include <map>\n#include <tuple>\n#include <cuda_runtime_api.h>\n#include <cuda.h>\n#include <cublas_v2.h>')
s=s.replace('struct Scope;','struct ApiStat {uint64_t count=0,bytes=0,errors=0;int64_t ns=0,max_ns=0;};\nstatic thread_local int in_cuda_graph=0,api_depth=0;\nstruct Scope;')
s=s.replace('json calls=json::array();','json calls=json::array();std::map<std::tuple<std::string,uintptr_t,int>,ApiStat>apis;\n json api_rows(){json v=json::array();for(auto&[key,c]:apis){auto&[name,pc,d]=key;Dl_info info{};dladdr((void*)pc,&info);v.push_back({{"kind",name},{"module",info.dli_fname?info.dli_fname:""},{"offset",pc-uintptr_t(info.dli_fbase)},{"depth",d},{"count",c.count},{"ns",c.ns},{"max_ns",c.max_ns},{"bytes",c.bytes},{"errors",c.errors}});}return v;}')
s=s.replace('{"calls",std::move(calls)}','{"calls",std::move(calls)},{"cuda_apis",api_rows()}')
start='auto rc=fn(b,g);c.j["status"]=int(rc);return rc;}'
repl='''bool gpu=ggml_backend_dev_type(ggml_backend_get_device(b))==GGML_BACKEND_DEVICE_TYPE_GPU;
if(gpu){json moe=json::array();for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(t->op==GGML_OP_MUL_MAT_ID||t->op==GGML_OP_MUL_MAT_ID_STAGED){auto*w=t->src[0];auto*x=t->src[1];moe.push_back({{"name",t->name},{"weights",w->name},{"type",ggml_type_name(w->type)},{"w_ne",{w->ne[0],w->ne[1],w->ne[2],w->ne[3]}},{"x_ne",{x->ne[0],x->ne[1],x->ne[2],x->ne[3]}}});}}c.j["moe"]=std::move(moe);++in_cuda_graph;}
auto rc=fn(b,g);if(gpu)--in_cuda_graph;c.j["status"]=int(rc);return rc;}'''
assert s.count(start)==1;s=s.replace(start,repl)
s+='''
struct ApiCall {
 Scope*s;std::tuple<std::string,uintptr_t,int>key;int64_t start=0;size_t bytes=0;
 ApiCall(const char*n,const void*pc,size_t nb):s(current&&in_cuda_graph?current:nullptr),bytes(nb){if(s){key={n,uintptr_t(pc),api_depth++};start=now_ns();}}
 void finish(int rc){if(s){auto dt=now_ns()-start;auto&c=s->apis[key];++c.count;c.ns+=dt;c.max_ns=std::max(c.max_ns,dt);c.bytes+=bytes;c.errors+=rc!=0;--api_depth;s=nullptr;}}
};
'''
specs=[
('cudaError_t','cudaStreamSynchronize','cudaStream_t stream','stream','0',None),
('cudaError_t','cudaDeviceSynchronize','void','','0',None),
('cudaError_t','cudaEventSynchronize','cudaEvent_t event','event','0',None),
('cudaError_t','cudaMemcpyAsync','void*dst,const void*src,size_t count,cudaMemcpyKind kind,cudaStream_t stream','dst,src,count,kind,stream','count','kind==cudaMemcpyDeviceToHost?"cudaMemcpyAsync.D2H":kind==cudaMemcpyHostToDevice?"cudaMemcpyAsync.H2D":"cudaMemcpyAsync.other"'),
('cudaError_t','cudaMemcpyPeerAsync','void*dst,int dstDev,const void*src,int srcDev,size_t count,cudaStream_t stream','dst,dstDev,src,srcDev,count,stream','count',None),
('cudaError_t','cudaLaunchKernelExC','const cudaLaunchConfig_t*config,const void*func,void**args','config,func,args','0',None),
('cudaError_t','cudaGraphInstantiate','cudaGraphExec_t*exec,cudaGraph_t graph,unsigned long long flags','exec,graph,flags','0',None),
('cudaError_t','cudaGraphExecUpdate','cudaGraphExec_t exec,cudaGraph_t graph,cudaGraphExecUpdateResultInfo*info','exec,graph,info','0',None),
('cudaError_t','cudaGraphExecDestroy','cudaGraphExec_t exec','exec','0',None),
('cudaError_t','cudaGraphDestroy','cudaGraph_t graph','graph','0',None),
('cudaError_t','cudaGraphLaunch','cudaGraphExec_t exec,cudaStream_t stream','exec,stream','0',None),
('cudaError_t','cudaMalloc','void**ptr,size_t count','ptr,count','count',None),
('cudaError_t','cudaFree','void*ptr','ptr','0',None),
('cudaError_t','cudaMemsetAsync','void*ptr,int val,size_t count,cudaStream_t stream','ptr,val,count,stream','count',None),
('cudaError_t','cudaFuncGetAttributes','cudaFuncAttributes*attr,const void*func','attr,func','0',None),
('cudaError_t','cudaFuncSetAttribute','const void*func,cudaFuncAttribute attr,int value','func,attr,value','0',None),
('cudaError_t','cudaGetDevice','int*dev','dev','0',None),
('cudaError_t','cudaSetDevice','int dev','dev','0',None),
('cudaError_t','cudaGetLastError','void','','0',None),
('cudaError_t','cudaPeekAtLastError','void','','0',None),
('CUresult','cuMemMap','CUdeviceptr ptr,size_t size,size_t offset,CUmemGenericAllocationHandle h,unsigned long long flags','ptr,size,offset,h,flags','size',None),
('CUresult','cuMemUnmap','CUdeviceptr ptr,size_t size','ptr,size','size',None),
('CUresult','cuMemCreate','CUmemGenericAllocationHandle*h,size_t size,const CUmemAllocationProp*prop,unsigned long long flags','h,size,prop,flags','size',None),
('CUresult','cuMemSetAccess','CUdeviceptr ptr,size_t size,const CUmemAccessDesc*desc,size_t count','ptr,size,desc,count','size',None),
('cublasStatus_t','cublasSetStream_v2','cublasHandle_t h,cudaStream_t stream','h,stream','0',None),
('cublasStatus_t','cublasSgemm_v2','cublasHandle_t h,cublasOperation_t a,cublasOperation_t b,int m,int n,int k,const float*alpha,const float*A,int lda,const float*B,int ldb,const float*beta,float*C,int ldc','h,a,b,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc','0',None),
('cublasStatus_t','cublasGemmEx','cublasHandle_t h,cublasOperation_t a,cublasOperation_t b,int m,int n,int k,const void*alpha,const void*A,cudaDataType_t at,int lda,const void*B,cudaDataType_t bt,int ldb,const void*beta,void*C,cudaDataType_t ct,int ldc,cublasComputeType_t compute,cublasGemmAlgo_t algo','h,a,b,m,n,k,alpha,A,at,lda,B,bt,ldb,beta,C,ct,ldc,compute,algo','0',None)
]
for ret,name,decl,args,nb,label in specs:
 label=label or '"'+name+'"';s+=f'extern "C" {ret} {name}({decl}){{static auto fn=next_fn<decltype(&{name})>("{name}");if(!current||!in_cuda_graph)return fn({args});ApiCall c({label},__builtin_return_address(0),{nb});auto rc=fn({args});c.finish(int(rc));return rc;}}\n'
s=s.replace('decltype(&cublasGemmEx)','cublasStatus_t(*)(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const void*,const void*,cudaDataType_t,int,const void*,cudaDataType_t,int,const void*,void*,cudaDataType_t,int,cublasComputeType_t,cublasGemmAlgo_t)')
(D/'probe.cpp').write_text(s)
av=['/usr/bin/c++','-O2','-std=c++17','-fPIC','-shared',str(D/'probe.cpp'),*[ '-I'+str(R/x)for x in ['src','include','common','ggml/src','ggml/include','vendor']],'-I/usr/local/cuda/include','-ldl','-pthread','-o',str(D/'probe.so')]
with(D/'build.log').open('w')as log:subprocess.run(av,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();(D/'build.json').write_text(json.dumps({'command':av,'source_sha256':sha(D/'probe.cpp'),'probe_sha256':sha(D/'probe.so'),'scope':'CUDA/driver/CUBLAS API counters inside existing GPU graph_compute only. Original call/arguments/status forwarded exactly; no extra GPU events, CUPTI, stack sampling, callback or math change. Phase1 remains no-CUDA-API-interposition host scopes.'},indent=2)+'\n')
s=(P/'run.py').read_text();s=s.replace("P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts'","P=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts/api-attribution'")
s=s.replace('19093','19094').replace("='prefill-residual'","='prefill-api'");s=s.replace('for n in [8192,98304]:','for n in [8192]:');s=s.replace('External host API scopes, forwards existing copy iface; no CUPTI/CUDA API interposition/sampling/callbacks/added GPU events or source/default changes. Pass-through controls before/after armed probe in same process. Default CPU lookup early remains unset/ON.','Phase2 host scopes plus bounded CUDA/driver/CUBLAS API counters inside GPU graph_compute. No CUPTI/sampling/callbacks/extra GPU events/math/source/default changes. Pass-through controls before/after one armed8K request, same process. CPU lookup early defaultON/unset.');(D/'run.py').write_text(s)
print('Bounded API attribution probe/controller built.')
