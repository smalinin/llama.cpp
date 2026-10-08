#!/usr/bin/env python3
"""Benchmark DSpark with fixed placement and separate diagnostic profiling."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error

ROOT = Path(__file__).resolve().parent.parent/'stage7'
REPO = Path('/home/sergei/Github/llama.cpp')
spec = importlib.util.spec_from_file_location('baseline', ROOT.parent/'run_baseline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SERVER = Path(__file__).resolve().parent.parent/'stage8/candidate-bin/llama-server'
GPU_ORDER = json.loads((ROOT.parent/'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order']
CONFIGS = {'off':None, 'n1-p0':(1,0), 'n2-p0':(2,0), 'n3-p0':(3,0),
           'n3-p03':(3,0.3), 'n3-p06':(3,0.6), 'k0':(3,0)}
QUESTIONS = {
    'svg':'Create a complete HTML page with an SVG of a pelican riding a bicycle. Use no external assets.',
    'python':'Write a Python function that merges overlapping intervals. Include type hints and three examples.',
    'explain':'Explain why a database index speeds up reads but can slow down writes. Give a concrete example.',
}
try:
    NVTX = ctypes.CDLL('libnvToolsExt.so.1')
    NVTX.nvtxRangePushA.argtypes = [ctypes.c_char_p]
except OSError:
    NVTX = None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(config, args):
    directory = args.output/config
    directory.mkdir(parents=True,exist_ok=False)
    profile = base.PROFILES['deepseek41']
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
    command = [str(SERVER),'-m',str(profile['model']),'-c','8192','-b','2048','-ub','512',
               '-np','1','-ctk','f16','-ctv','f16','-ctkd','f16','-ctvd','f16',
               '-t','12','-tb','12','-fit','off','-ngl','99','--tensor-split','1,1,1,1,1,0.4',
               '--split-mode','layer','--load-mode','none','--lazy-mode','auto',
               '--host','127.0.0.1','--port',str(port),'--metrics','--jinja','--verbosity',str(args.verbosity)]
    if CONFIGS[config]:
        n,p = CONFIGS[config]
        command += ['--spec-type','draft-dspark','--spec-draft-n-max',str(n),
                    '--spec-draft-p-min',str(p),'--model-draft',str(profile['draft']),
                    '--spec-draft-ngl','99']
    if config == 'k0':
        command += ['--spec-verify-policy','fixed','--spec-verify-k','0']
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):
            env.pop(key)
    for key in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION',
                'LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE',
                'GGML_CUDA_CUBLAS_COMPUTE_TYPE'):
        env.pop(key,None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=GPU_ORDER,LD_LIBRARY_PATH=str(SERVER.parent))
    if args.profile: env['LLAMA_DSPARK_PROFILE']='1'
    if args.tf32_off: env['NVIDIA_TF32_OVERRIDE']='0'
    hashes = json.loads((Path(__file__).resolve().parent.parent/'stage8/candidate-binary-sha256.json').read_text())
    for name,value in hashes.items():
        assert sha(SERVER.parent/name)==value,name
    manifest = {'command':command,'config':config,'gpu_order':GPU_ORDER,'ld_library_path':str(SERVER.parent),
                'tested_source_commit':'11638b68545860e96b055798e995bb14be3d0e88',
                'snapshot':'stage8/candidate-bin','pending_continue_fix_included':True,
                'pending_fp32_fix_included':True,
                'profile':args.profile,'tf32_off':args.tf32_off,'nvtx_available':NVTX is not None,
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'tensor_split':'1,1,1,1,1,0.4','fit':'off','log_verbosity':args.verbosity,
                'diagnostic_repeats':args.diagnostic_repeats}
    base.save(directory/'manifest.json',manifest)
    peaks = {'started':time.monotonic(),'gpu_mib':{},'process_hwm_kib':0}
    stop = threading.Event()
    result = {'config':config,'status':'running','requests':[]}
    print('START',config,flush=True)
    with (directory/'server.log').open('w') as log:
        proc = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        watcher = threading.Thread(target=base.monitor,args=(proc,directory,stop,peaks),daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic()+900
            while True:
                if proc.poll() is not None: raise RuntimeError('startup exit '+str(proc.returncode))
                try:
                    if base.api(port,'/health',timeout=3).get('status')=='ok': break
                except (urllib.error.URLError,TimeoutError): pass
                if time.monotonic()>deadline: raise TimeoutError('startup timeout')
                time.sleep(1)
            result['startup_s']=time.monotonic()-peaks['started']
            maps=Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            libs=sorted({line.split()[-1] for line in maps if any('/'+prefix in line for prefix in ('libllama','libggml','libmtmd'))})
            assert libs and all(Path(p).parent.resolve()==SERVER.parent.resolve() for p in libs),libs
            manifest['loaded_libraries']={p:sha(p) for p in libs}
            assert all(hashes[Path(p).name]==v for p,v in manifest['loaded_libraries'].items())
            props=base.api(port,'/props')
            base.save(directory/'props.json',props)
            manifest['embedded_build_info']=props['build_info']
            base.save(directory/'manifest.json',manifest)
            print('READY',config,round(result['startup_s'],1),flush=True)
            prompts={'baseline':base.PROMPT}
            for name,text in QUESTIONS.items():
                request={'messages':[{'role':'user','content':text}],
                         'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
                response=base.api(port,'/apply-template',request)
                if not isinstance(response['prompt'],str): raise RuntimeError('expected template text')
                prompts[name]=response['prompt']
                base.save(directory/(name+'-template-request.json'),request)
                (directory/(name+'-prompt.txt')).write_text(response['prompt'])
            def request(name,phase,repeat,diagnostic=False):
                label=f'{name}-{phase}-{repeat}'
                payload={'prompt':prompts[name],'n_predict':128 if diagnostic else 256,
                         'temperature':0,'seed':1234,'top_k':40,'top_p':0.95,'min_p':0.05,
                         'cache_prompt':False,'return_tokens':True,'stream':False}
                if diagnostic: payload['n_probs']=5
                base.save(directory/(label+'-request.json'),payload)
                if NVTX: NVTX.nvtxRangePushA(label.encode())
                offset=(directory/'server.log').stat().st_size
                started=time.monotonic()
                try: response=base.api(port,'/completion',payload,timeout=600)
                finally:
                    if NVTX: NVTX.nvtxRangePop()
                elapsed=time.monotonic()-started
                if args.profile: time.sleep(0.05)
                base.save(directory/(label+'-response.json'),response)
                assert 'error' not in response and response.get('tokens'),response
                assert len(response['tokens'])==response['tokens_predicted']
                assert response['timings'].get('cache_n',0)==0
                item={'request':label,'prompt':name,'phase':phase,'repeat':repeat,'wall_s':elapsed,
                      'timings':response['timings'],'tokens_predicted':response['tokens_predicted'],
                      'content_sha256':hashlib.sha256(response['content'].encode()).hexdigest(),
                      'tokens_sha256':hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest(),
                      'log_start_offset':offset,'log_end_offset':(directory/'server.log').stat().st_size}
                result['requests'].append(item)
                base.save(directory/'result.json',result)
                print('DONE',config,label,json.dumps(response['timings']),flush=True)
            for repeat in range(1,args.diagnostic_repeats+1): request('baseline','diagnostic',repeat,True)
            if not args.diagnostic_only:
                for name in prompts: request(name,'warmup',0)
                for repeat in range(1,4):
                    names=list(prompts)
                    names=names[repeat-1:]+names[:repeat-1]
                    for name in names: request(name,'measure',repeat)
            result['status']='passed'
        except Exception as error:
            result.update(status='failed',error=repr(error))
        finally:
            stop.set();watcher.join(timeout=15)
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            text=(directory/'server.log').read_text(errors='replace')
            manifest['model_buffers']=re.findall(r'load_tensors:\s+[^\n]*model buffer size =[^\n]+',text)
            manifest['compute_buffers']=re.findall(r'[^\n]*compute buffer size =[^\n]+',text)
            manifest['spec_init']=re.findall(r'[^\n]*common_speculative_impl_draft_dflash[^\n]*',text)
            manifest['memory_rm_type']=re.findall(r'[^\n]*(?:seq_rm|rollback|checkpoint)[^\n]*',text)[:25]
            base.save(directory/'manifest.json',manifest)
            result.update(server_exit_code=proc.returncode,total_s=time.monotonic()-peaks['started'],
                          peak_gpu_used_mib=peaks['gpu_mib'],peak_process_rss_kib=peaks['process_hwm_kib'])
            base.save(directory/'result.json',result)
    print('FINISH',config,result['status'],result.get('error',''),flush=True)
    if result['status']!='passed':raise RuntimeError(result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--configs',nargs='+',choices=CONFIGS,default=['off','n1-p0','n2-p0','n3-p0','n3-p03','n3-p06'])
    parser.add_argument('--output',type=Path,default=ROOT/'runs')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--diagnostic-only',action='store_true')
    parser.add_argument('--tf32-off',action='store_true')
    parser.add_argument('--verbosity',type=int,default=3)
    parser.add_argument('--diagnostic-repeats',type=int,default=2)
    args=parser.parse_args()
    results=[]
    for config in args.configs:
        results.append(run(config,args))
        base.save(args.output/'results.json',results)
