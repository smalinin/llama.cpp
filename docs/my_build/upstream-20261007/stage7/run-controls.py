#!/usr/bin/env python3
"""Run GPU controls only after the main benchmark has stopped."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/sergei/Github/llama.cpp')
SNAPSHOT = ROOT.parent/'stage5/candidate-bin'
NSYS = '/opt/nvidia/nsight-systems/2025.3.2/bin/nsys'

def run(label, command, env=None):
    print('START',label,flush=True)
    with (ROOT/(label+'.log')).open('w') as output:
        subprocess.run(command,cwd=ROOT.parent.parent,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    print('FINISH',label,flush=True)

def replay():
    model = json.loads((ROOT/'model-inspection.json').read_text())['models']['model']['path']
    order = json.loads((ROOT/'runs/off/manifest.json').read_text())['gpu_order']
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):
            env.pop(key)
    for key in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION',
                'LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE'):
        env.pop(key,None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=order,LD_LIBRARY_PATH=str(SNAPSHOT))
    native = json.loads((ROOT/'runs/off/baseline-diagnostic-1-response.json').read_text())['tokens']
    assert len(native)==128
    (ROOT/'forced-native.i32').write_bytes(struct.pack('<128i',*native))
    command = [str(ROOT/'target-replay-server-matched'),model,str(ROOT/'baseline-prompt.txt'),
               str(ROOT/'forced-native.i32'),str(ROOT/'target-replay-server-matched-output')]
    manifest = {'command':command,'gpu_order':order,'ld_library_path':str(SNAPSHOT),
                'binary_sha256':hashlib.sha256((ROOT/'target-replay-server-matched').read_bytes()).hexdigest(),
                'source_sha256':hashlib.sha256((ROOT/'target-replay.cpp').read_bytes()).hexdigest(),
                'tf32_override':None,'swa_full':False,'no_perf':False,'prefill_chunks':[1139,512,4]}
    (ROOT/'target-replay-server-matched-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    run('target-replay-server-matched',command,env)
    run('target-replay-server-matched-analysis',[sys.executable,str(ROOT/'analyze.py'),'replay',str(ROOT/'target-replay-server-matched-output')])

def nsight():
    env = os.environ.copy()
    env['NSYS_NVTX_PROFILER_REGISTER_ONLY']='0'
    command = [NSYS,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
               '--capture-range=nvtx','--nvtx-capture=baseline-diagnostic-2',
               '--capture-range-end=stop','--kill=none','--wait=all','--cuda-graph-trace=node',
               '--output',str(ROOT/'nsys-dspark'),sys.executable,'-u',str(ROOT/'run-dspark.py'),
               '--configs','n3-p0','--profile','--diagnostic-only','--verbosity','4',
               '--diagnostic-repeats','3',
               '--output',str(ROOT/'profile-runs')]
    (ROOT/'nsys-command.json').write_text(json.dumps(command,indent=2)+'\n')
    run('nsys',command,env)
    run('nsys-export',[NSYS,'export','--type','sqlite','--output',str(ROOT/'nsys-dspark.sqlite'),str(ROOT/'nsys-dspark.nsys-rep')])
    run('profile-analysis',[sys.executable,str(ROOT/'analyze.py'),'profile',str(ROOT/'profile-runs')])

def k0():
    run('k0',[sys.executable,'-u',str(ROOT/'run-dspark.py'),'--configs','k0','--profile',
              '--diagnostic-only','--verbosity','4','--output',str(ROOT/'k0-runs')])
    run('k0-analysis',[sys.executable,str(ROOT/'analyze.py'),'profile',str(ROOT/'k0-runs')])

if __name__ == '__main__':
    for result in json.loads((ROOT/'runs/results.json').read_text()):
        assert result['status']=='passed' and result['server_exit_code']==0
    assert len(json.loads((ROOT/'runs/results.json').read_text()))==6
    replay()
    nsight()
    k0()
