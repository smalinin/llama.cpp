#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>
#include <dlfcn.h>
#include <execinfo.h>

static std::atomic<bool> active{false}, done{false};
static std::atomic<int> phase{0}, capture_id{0};
static std::mutex log_mu;
static FILE * logs = nullptr;
static int listener = -1, peer = -1;
static std::thread worker;
static long long stamp() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000000000LL+t.tv_nsec; }
static void entry(const char * action) { std::lock_guard<std::mutex> lock(log_mu); if(logs){fprintf(logs,"{\"event\":\"control\",\"action\":\"%s\",\"capture\":%d,\"phase\":%d,\"ns\":%lld,\"pid\":%d}\n",action,capture_id.load(),phase.load(),stamp(),int(getpid()));fflush(logs);} }
static void sample(const char * api, long long start, long long end) {
 void * frames[32]; int n=backtrace(frames,32);
 std::lock_guard<std::mutex> lock(log_mu); if(!logs)return;
 fprintf(logs,"{\"event\":\"stack\",\"api\":\"%s\",\"capture\":%d,\"phase\":%d,\"tid\":%ld,\"start\":%lld,\"end\":%lld,\"frames\":[",api,capture_id.load(),phase.load(),syscall(SYS_gettid),start,end);
 for(int i=1;i<n;++i){Dl_info x{};dladdr(frames[i],&x);fprintf(logs,"%s{\"file\":\"%s\",\"offset\":%llu}",i>1?",":"",x.dli_fname?x.dli_fname:"",(unsigned long long)((uintptr_t)frames[i]-(uintptr_t)x.dli_fbase));}
 fprintf(logs,"]}\n");fflush(logs);
}
static bool selected(int kind, unsigned period) {
 if(!active.load(std::memory_order_relaxed))return false;
 thread_local int previous=-1; thread_local unsigned counts[5]{};
 int id=capture_id.load(); if(previous!=id){previous=id;memset(counts,0,sizeof counts);}
 return counts[kind]++%period==0;
}
template<class F> static F next(const char * name){void * p=dlsym(RTLD_NEXT,name);if(!p){fprintf(stderr,"probe missing %s\n",name);abort();}return reinterpret_cast<F>(p);}
extern "C" bool dsv_profile_active(){return active.load();}
extern "C" int dsv_profile_phase(){return phase.load();}
extern "C" cudaError_t cudaLaunchKernel(const void*f,dim3 g,dim3 b,void**a,size_t s,cudaStream_t st){using F=cudaError_t(*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);static auto fn=next<F>("cudaLaunchKernel");bool yes=selected(0,64);long long t=yes?stamp():0;auto r=fn(f,g,b,a,s,st);if(yes)sample("cudaLaunchKernel",t,stamp());return r;}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t f,dim3 g,dim3 b,void**a,size_t s,cudaStream_t st){using F=cudaError_t(*)(cudaKernel_t,dim3,dim3,void**,size_t,cudaStream_t);static auto fn=next<F>("__cudaLaunchKernel");bool yes=selected(0,64);long long t=yes?stamp():0;auto r=fn(f,g,b,a,s,st);if(yes)sample("__cudaLaunchKernel",t,stamp());return r;}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t*c,const void*f,void**a){using F=cudaError_t(*)(const cudaLaunchConfig_t*,const void*,void**);static auto fn=next<F>("cudaLaunchKernelExC");bool yes=selected(0,64);long long t=yes?stamp():0;auto r=fn(c,f,a);if(yes)sample("cudaLaunchKernelExC",t,stamp());return r;}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g,cudaStream_t s){using F=cudaError_t(*)(cudaGraphExec_t,cudaStream_t);static auto fn=next<F>("cudaGraphLaunch");bool yes=selected(1,16);long long t=yes?stamp():0;auto r=fn(g,s);if(yes)sample("cudaGraphLaunch",t,stamp());return r;}
extern "C" cudaError_t cudaMemcpyAsync(void*d,const void*s,size_t n,cudaMemcpyKind k,cudaStream_t st){using F=cudaError_t(*)(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t);static auto fn=next<F>("cudaMemcpyAsync");bool yes=selected(2,64);long long t=yes?stamp():0;auto r=fn(d,s,n,k,st);if(yes)sample("cudaMemcpyAsync",t,stamp());return r;}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t s){using F=cudaError_t(*)(cudaStream_t);static auto fn=next<F>("cudaStreamSynchronize");bool yes=selected(3,16);long long t=yes?stamp():0;auto r=fn(s);if(yes)sample("cudaStreamSynchronize",t,stamp());return r;}
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t s){using F=cudaError_t(*)(cudaEvent_t);static auto fn=next<F>("cudaEventSynchronize");bool yes=selected(4,16);long long t=yes?stamp():0;auto r=fn(s);if(yes)sample("cudaEventSynchronize",t,stamp());return r;}
__attribute__((constructor)) static void setup(){
 const char * path=getenv("DSV_PROFILE_SOCKET"), * file=getenv("DSV_PROFILE_LOG");if(!path||!file)return;
 logs=fopen(file,"a"); if(!logs)abort();
 listener=socket(AF_UNIX,SOCK_STREAM,0);sockaddr_un addr{};addr.sun_family=AF_UNIX;if(strlen(path)>=sizeof addr.sun_path)abort();strcpy(addr.sun_path,path);unlink(path);if(bind(listener,(sockaddr*)&addr,sizeof addr)||listen(listener,1))abort();
 if(const char * pid=getenv("DSV_PROFILE_PID")){FILE*f=fopen(pid,"w");if(f){fprintf(f,"%d\n",int(getpid()));fclose(f);}}
 worker=std::thread([]{
  peer=accept(listener,nullptr,nullptr);if(peer<0)return;FILE * io=fdopen(peer,"r+");if(!io)return;char line[128];nvtxRangeId_t range=0;
  while(!done.load()&&fgets(line,sizeof line,io)){
   int id=0;
   if(sscanf(line,"START %d",&id)==1){capture_id=id;phase=1;cudaProfilerStart();range=nvtxRangeStartA("phase.prefill");active=true;entry("start");}
   else if(!strncmp(line,"TG",2)){if(range)nvtxRangeEnd(range);phase=2;range=nvtxRangeStartA("phase.generation");entry("first_output");}
   else if(!strncmp(line,"STOP",4)){active=false;if(range)nvtxRangeEnd(range);range=nvtxRangeStartA("phase.drain_after_response");phase=3;int n=0;cudaGetDeviceCount(&n);for(int d=0;d<n;++d){cudaSetDevice(d);auto rc=cudaDeviceSynchronize();if(rc!=cudaSuccess)abort();}nvtxRangeEnd(range);range=0;cudaProfilerStop();entry("stop");phase=0;}
   fprintf(io,"ACK %d %d %lld\n",capture_id.load(),phase.load(),stamp());fflush(io);
  }
  fclose(io);peer=-1;
 });
}
__attribute__((destructor)) static void finish(){done=true;if(peer>=0)shutdown(peer,SHUT_RDWR);if(listener>=0){shutdown(listener,SHUT_RDWR);close(listener);}if(worker.joinable())worker.join();if(logs)fclose(logs);}
