#!/usr/bin/env python3
"""Paired real-model checks on the post-replacement hardware."""
import argparse,csv,json
from pathlib import Path
ROOT=Path('/home/sergei/_my_sync/llama_upstream_review');R=ROOT/'stage4'
rows=list(csv.DictReader((R/'gpus.csv').read_text().splitlines()))
uuids={int(row['index']):row[' uuid'].strip() for row in rows}
source=(ROOT/'run_baseline.py').read_text()
source=source.replace("GPU_ORDER = '2,1,0,5,4,3'", "GPU_ORDER = "+repr(','.join(uuids[i] for i in (2,1,0,5,4,3))))
source=source.replace("env = os.environ.copy()", "env = os.environ.copy()\n    env['LD_LIBRARY_PATH'] = str(server.parent)\n    for option in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION'):\n        env.pop(option,None)")
source=source.replace("'--metrics',", "'--metrics', '--verbosity', '5',")
source=source.replace("    if mtp:\n        command +=", "    if name in ('glm5next','qwen4exp'):\n        i=command.index('-fit')\n        command[i:i+4]=['-fit','off','--tensor-split','1,1,1,1,1,0.4','-ngl','99']\n    if mtp:\n        command +=")
probe="""            maps = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            libs = sorted({line.split()[-1] for line in maps if any('/'+prefix in line for prefix in ('libllama.so.','libllama-server-impl.so','libggml-cuda.so.','libggml-base.so.'))})
            if len(libs)!=4 or any(Path(f).parent.resolve()!=server.parent.resolve() for f in libs):
                raise RuntimeError('unexpected library paths: '+repr(libs))
            manifest['loaded_libraries']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in libs}
            manifest['ld_library_path']=env['LD_LIBRARY_PATH']
            manifest['hardware_date']='2026-10-08'
            manifest['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            manifest['binary_set']=server.parent.name
            log_text=(directory/'server.log').read_text(errors='replace')
            import re
            placement=re.findall(r'load_tensors: layer\\s+\\d+ assigned to device [^\\n]+',log_text)
            buffers=re.findall(r'load_tensors:\\s+[^\\n]*model buffer size =[^\\n]+',log_text)
            manifest['layer_placement']=placement
            manifest['model_buffers']=buffers
            if not placement or not buffers:raise RuntimeError('missing placement evidence')
            save(directory/'manifest.json',manifest)
"""
source=source.replace("            result['startup_s'] =",probe+"            result['startup_s'] =")
extra="""            import re
            log_text=(directory/'server.log').read_text(errors='replace')
            result['layer_placement']=re.findall(r'load_tensors: layer\\s+\\d+ assigned to device [^\\n]+',log_text)
            if name in ('qwen4exp','glm5next'):
                long_prompt='Read the following numbered facts.\\n'+''.join(f'Entry {i}: the checkpoint value is {1000+i}; the label is blue.\\n' for i in range(300))+'\\nExplain how to check these entries, then list the first ten checkpoint values.\\nAnswer:\\n'
                for request_name,prompt in [('greedy-3',PROMPT),('greedy-4',PROMPT)]+[(f'long-greedy-{i}',long_prompt) for i in range(1,5)]:
                    payload={'prompt':prompt,'n_predict':128,'temperature':0,'seed':1234,'top_k':40,'top_p':.95,'min_p':.05,'cache_prompt':False,'return_tokens':True,'stream':False}
                    save(directory/(request_name+'-request.json'),payload)
                    started=time.monotonic();response=api(port,'/completion',payload)
                    save(directory/(request_name+'-response.json'),response)
                    if 'error' in response or not response.get('tokens'):raise RuntimeError('extra request failed')
                    item={'request':request_name,'wall_s':time.monotonic()-started,'timings':response.get('timings'),'tokens_predicted':response.get('tokens_predicted'),'content_sha256':hashlib.sha256(response.get('content','').encode()).hexdigest(),'tokens_sha256':hashlib.sha256(json.dumps(response.get('tokens')).encode()).hexdigest()}
                    result['requests'].append(item);save(directory/'result.json',result)
                    print(f'DONE {directory.name} {request_name}: {json.dumps(item["timings"])}',flush=True)
"""
source=source.replace("            result['status'] = 'passed'",extra+"            result['status'] = 'passed'")
namespace={'__name__':'stage4_real'};exec(compile(source,str(ROOT/'run_baseline.py'),'exec'),namespace)
parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=list(namespace['PROFILES']),choices=namespace['PROFILES']);args=parser.parse_args()
results=[]
for name in args.models:
    for spec in (False,True):
        for version in ('before','after'):
            out=R/('real-'+version);out.mkdir(exist_ok=True)
            result=namespace['run_case'](name,spec,out,R/(version+'-bin/llama-server'))
            result['binary_set']=version;results.append(result)
            namespace['save'](R/'real-progress.json',results)
            if result['status']!='passed':raise SystemExit(1)
