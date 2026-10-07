#!/usr/bin/env python3
"""Real-model baseline requests and slot state save/restore/failure checks."""
import json
from pathlib import Path
ROOT=Path('/home/sergei/_my_sync/llama_upstream_review')
source=(ROOT/'run_baseline.py').read_text()
source=source.replace("env = os.environ.copy()", "env = os.environ.copy()\n    env['LD_LIBRARY_PATH'] = str(server.parent)")
source=source.replace("    profile = PROFILES[name]", "    profile = PROFILES[name]\n    (directory/'slots').mkdir()")
source=source.replace("'--metrics',", "'--metrics', '--slot-save-path', str(directory/'slots')+'/',")
source=source.replace("    if mtp:\n        command +=", "    if name == 'glm5next':\n        i=command.index('-fit')\n        command[i:i+4]=['-fit','off','--tensor-split','0.9,0.9,0.9,0.4,0.9,0.3','-ngl','99']\n    if name == 'qwen4exp' and os.getenv('STAGE3_QWEN_Q8'):\n        command[command.index('-ctk')+1]='q8_0'\n        command[command.index('-ctv')+1]='q8_0'\n    if mtp:\n        command +=")
probe="""            maps = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            libs = sorted({line.split()[-1] for line in maps if '/libllama.so.' in line or '/libllama-server-impl.so' in line})
            if len(libs)!=2 or any(Path(f).parent.resolve()!=server.parent.resolve() for f in libs):
                raise RuntimeError('unexpected library paths: '+repr(libs))
            manifest['loaded_libraries']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in libs}
            manifest['ld_library_path']=env['LD_LIBRARY_PATH']
            save(directory/'manifest.json',manifest)
"""
source=source.replace("            result['startup_s'] =",probe+"            result['startup_s'] =")
extra="""            import struct
            import shutil
            slot_path=directory/'slots'
            def operation(label,action,filename=None,expect_error=False):
                request={} if filename is None else {'filename':filename}
                save(directory/(label+'-request.json'),request)
                try:
                    response=api(port,'/slots/0?action='+action,request)
                    status=200
                except urllib.error.HTTPError as error:
                    status=error.code
                    response=json.loads(error.read())
                save(directory/(label+'-response.json'),{'status':status,'body':response})
                if expect_error:
                    if status!=400: raise RuntimeError('expected restore rejection: '+repr(response))
                elif status!=200 or 'error' in response:
                    raise RuntimeError('slot action failed: '+repr(response))
                return {'label':label,'status':status,'response':response}
            result['slot_tests']=[]
            result['slot_tests'].append(operation('slot-save','save','good.bin'))
            result['slot_tests'].append(operation('slot-erase','erase'))
            result['slot_tests'].append(operation('slot-restore','restore','good.bin'))
            result['slot_tests'].append(operation('slot-resave','save','resaved.bin'))
            good=slot_path/'good.bin'
            original_hash=hashlib.sha256(good.read_bytes()).hexdigest()
            restored_hash=hashlib.sha256((slot_path/'resaved.bin').read_bytes()).hexdigest()
            result['state_roundtrip_bitwise_equal']=original_hash==restored_hash
            if not result['state_roundtrip_bitwise_equal']: raise RuntimeError('slot state roundtrip mismatch')
            for label in ('restored','truncated','old-version'):
                if label!='restored':
                    bad=slot_path/(label+'.bin')
                    shutil.copyfile(good,bad)
                    with bad.open('r+b') as file:
                        if label=='truncated': file.truncate(good.stat().st_size-1)
                        else: file.seek(4);file.write(struct.pack('<I',3))
                    result['slot_tests'].append(operation('slot-'+label,'restore',bad.name,True))
                request={'prompt':PROMPT,'n_predict':128,'temperature':0,'seed':1234,'top_k':40,'top_p':.95,'min_p':.05,
                         'cache_prompt':label=='restored','return_tokens':True,'stream':False}
                save(directory/('state-'+label+'-request.json'),request)
                response=api(port,'/completion',request)
                save(directory/('state-'+label+'-response.json'),response)
                if 'error' in response or not response.get('tokens'): raise RuntimeError('request after restore failed')
                result['requests'].append({'request':'state-'+label,'timings':response.get('timings'),
                                           'tokens_predicted':response.get('tokens_predicted')})
            result['slot_state']={'bytes':good.stat().st_size,'sha256':original_hash,'resaved_sha256':restored_hash,
                                  'file_version':struct.unpack('<I',good.read_bytes()[4:8])[0]}
"""
extra += """            if os.getenv('STAGE3_CONTINUATION'):
                continuation_namespace={}
                exec(compile((Path('/home/sergei/_my_sync/llama_upstream_review/stage3/check-continuation.py')).read_text(),'check-continuation.py','exec'),continuation_namespace)
                continuation_namespace['check_continuation'](api,port,directory,result,PROMPT,allow_full_prefill=mtp)
"""
source=source.replace("            result['status'] = 'passed'",extra+"            result['status'] = 'passed'")
namespace={'__name__':'real_state_check'}
exec(compile(source,str(ROOT/'run_baseline.py'),'exec'),namespace)
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--qwen-q8',action='store_true');args=parser.parse_args()
import os
if args.qwen_q8:
    os.environ['STAGE3_QWEN_Q8']='1'
    os.environ['STAGE3_CONTINUATION']='1'
out=ROOT/'stage3'/('runs-q8' if args.qwen_q8 else 'runs');out.mkdir(exist_ok=False)
results=[]
for name in (['qwen4exp'] if args.qwen_q8 else ['glm5next','qwen4exp','glm-dsa','deepseek41']):
    results.append(namespace['run_case'](name,True,out,ROOT/'stage3/bin/llama-server'))
    namespace['save'](out/'summary.json',results)
raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)
