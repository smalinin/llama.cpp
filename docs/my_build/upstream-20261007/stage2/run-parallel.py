#!/usr/bin/env python3
"""Serial, concurrent and cache-reuse requests in three server slots."""
import argparse
import json
from pathlib import Path

ROOT=Path('/home/sergei/_my_sync/llama_upstream_review')
source=(ROOT/'run_baseline.py').read_text()
source=source.replace("env = os.environ.copy()", "env = os.environ.copy()\n    env['LD_LIBRARY_PATH'] = str(server.parent) + (os.pathsep + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')")
probe="""            maps = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            libraries = sorted({line.split()[-1] for line in maps
                                if '/libllama.so.' in line or '/libllama-server-impl.so' in line})
            if len(libraries) != 2 or any(Path(path).parent.resolve() != server.parent.resolve() for path in libraries):
                raise RuntimeError('unexpected llama library paths: ' + repr(libraries))
            manifest['loaded_libraries'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in libraries}
            manifest['ld_library_path'] = env['LD_LIBRARY_PATH']
            save(directory / 'manifest.json', manifest)
"""
source=source.replace("            result['startup_s'] =", probe + "            result['startup_s'] =")
source=source.replace("'-ub', '512', '-np', '1'", "'-ub', '64', '-np', '3'")
source=source.replace("'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),", "'prompts': 'See per-slot request JSON files.',")
source=source.replace("(directory / 'prompt.txt').write_text(PROMPT)", "(directory / 'prompt.txt').write_text('See per-slot request JSON files.\\n')")
start=source.index('            for request_name, temperature in ')
end=source.index('        except Exception as error:',start)
source=source[:start]+'''            from concurrent.futures import ThreadPoolExecutor
            prompts = [
                'Explain in five short steps how to check numbered records for missing entries and duplicate values. Answer:\\n',
                'Write a Python function that returns duplicate integers from a list. Add a short example. Answer:\\n',
                'List the first twelve prime numbers, then explain how to test a number for primality. Answer:\\n',
            ]
            def request(phase, index):
                payload = {'prompt': prompts[index], 'n_predict': 64, 'temperature': 0,
                           'seed': 1234, 'top_k': 40, 'top_p': .95, 'min_p': .05,
                           'cache_prompt': phase == 'reuse', 'return_tokens': True,
                           'id_slot': index}
                save(directory / f'{phase}-{index}-request.json', payload)
                started = time.monotonic()
                response = api(port, '/completion', payload)
                save(directory / f'{phase}-{index}-response.json', response)
                if 'error' in response or not response.get('tokens'):
                    raise RuntimeError(str(response))
                return {'phase': phase, 'slot': index, 'wall_s': time.monotonic() - started,
                        'timings': response.get('timings'), 'stop_type': response.get('stop_type'),
                        'tokens': len(response['tokens'])}
            for index in range(3):
                result['requests'].append(request('serial', index))
            for phase in ('parallel1', 'parallel2', 'reuse'):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    result['requests'].extend(pool.map(lambda i: request(phase, i), range(3)))
            result['comparisons'] = []
            for index in range(3):
                responses = {phase: json.loads((directory / f'{phase}-{index}-response.json').read_text())
                             for phase in ('serial', 'parallel1', 'parallel2', 'reuse')}
                result['comparisons'].append({
                    'slot': index,
                    'serial_parallel_equal': responses['serial']['tokens'] == responses['parallel1']['tokens'],
                    'parallel_repeats_equal': responses['parallel1']['tokens'] == responses['parallel2']['tokens'],
                    'reuse_parallel_equal': responses['parallel2']['tokens'] == responses['reuse']['tokens'],
                })
            result['status'] = 'passed'
''' + source[end:]


def run(label, name, kind, draft, unified, before=False):
    namespace={'__name__': 'parallel_check'}
    code=source.replace("'--metrics',", "'--metrics', " + repr('--kv-unified' if unified else '--no-kv-unified') + ',')
    exec(compile(code,str(ROOT/'run_baseline.py'),'exec'),namespace)
    profile=namespace['PROFILES'][name]
    profile['spec_type']=kind
    if draft: profile['draft']=Path(draft)
    out=ROOT/'stage2/parallel'/label
    out.mkdir(parents=True,exist_ok=False)
    server=ROOT/('stage1' if before else 'stage2')/'bin/llama-server'
    result=namespace['run_case'](name,True,out,server)
    result.update({'label':label,'before':before,'kv_unified':unified})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');args=parser.parse_args()
    dflash='/home/sergei/.models/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/GLM-5.3-Flash-DFlash2-Q8_0.gguf'
    profiles=[('dflash-separated-before' if args.before else 'dflash-separated','glm5next','draft-dflash',dflash,False)]
    if args.before:
        profiles += [('dflash-unified-before','glm5next','draft-dflash',dflash,True)]
    if not args.before:
        profiles += [('dflash-unified','glm5next','draft-dflash',dflash,True),
                     ('glm-mtp-separated','glm5next','draft-mtp',None,False),
                     ('qwen-mtp-separated','qwen4exp','draft-mtp',None,False)]
    results=[]
    for profile in profiles:
        results.append(run(*profile,before=args.before))
        path=ROOT/'stage2'/('parallel-before-summary.json' if args.before else 'parallel-summary.json')
        path.write_text(json.dumps(results,indent=2)+'\n')
    raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)


if __name__=='__main__': main()
