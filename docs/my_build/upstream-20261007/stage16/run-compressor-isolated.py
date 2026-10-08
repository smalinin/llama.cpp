import hashlib,json,os,subprocess
from pathlib import Path

R=Path(__file__).resolve().parent
SNAP=R.parent/'stage8/candidate-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
assert all(sha(SNAP/n)==h for n,h in hashes.items())
order=json.loads((R.parent/'stage7/runs/off/manifest.json').read_text())['gpu_order'].split(',')
env=os.environ.copy()
for k in list(env):
    if k.startswith(('LLAMA_ARG_','LLAMA_MTP_','LLAMA_DSPARK_','GGML_SCHED_')):env.pop(k)
for k in ['GGML_CUDA_DISABLE_FUSION','LLAMA_FUSED_LID_DISABLE','QWEN4EXP_FUSED_LID','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE']:env.pop(k,None)
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',LD_LIBRARY_PATH=str(SNAP),GGML_CUDA_DISABLE_GRAPHS='1')
meta={'head':subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip(),
      'snapshot_hashes_verified':len(hashes),'production_change':False,'runs':[]}
for arch,gpu in [('ada',order[0]),('ampere',order[-1])]:
    env['CUDA_VISIBLE_DEVICES']=gpu
    for dataset in ['kv','gate','w1','w2','w4','w2-swap']:
        compressor=dataset in ['kv','gate']
        binary=R/('compressor-replay' if compressor else 'attention-replay')
        inputs=R/('compressor-inputs' if compressor else 'attention-inputs')/dataset
        output=R/f'{"compressor" if compressor else "attention"}-{dataset}-{arch}'
        command=[str(binary),str(inputs)]+(['projection'] if compressor else [])+[str(output)]
        record={'architecture':arch,'gpu':gpu,'dataset':dataset,'command':command,'binary_sha256':sha(binary),
                'source_sha256':sha(binary.with_suffix('.cpp')),'input_manifest_sha256':sha(inputs/'manifest.json')}
        print('START',output.name,flush=True)
        with (R/(output.name+'.log')).open('w') as log:proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['exit_code']=proc.returncode;meta['runs'].append(record)
        (R/'compressor-isolated-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        assert proc.returncode==0
        loaded={line.split()[-1] for line in (output/'loaded-libraries.txt').read_text().splitlines()}
        assert loaded and all(Path(p).parent==SNAP and sha(p)==hashes[Path(p).name] for p in loaded)
        record['loaded_libraries']={p:sha(p) for p in sorted(loaded)}
        (R/'compressor-isolated-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('FINISH',output.name,flush=True)
