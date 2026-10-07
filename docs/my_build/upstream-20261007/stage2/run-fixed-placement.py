#!/usr/bin/env python3
"""Compare the same GLM5NEXT requests with fixed GPU layer fractions and fit disabled."""
import argparse
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
source=source.replace("'-fit', 'on', '--fit-target', '3072'", "'-fit', 'off', '--tensor-split', '0.9,0.9,0.9,0.4,0.9,0.3', '-ngl', '99'")
parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');args=parser.parse_args()
namespace={'__name__':'fixed_placement'}
exec(compile(source,str(ROOT/'run_baseline.py'),'exec'),namespace)
label='fixed-before' if args.before else 'fixed-after'
out=ROOT/'stage2'/label;out.mkdir(exist_ok=False)
server=ROOT/('stage1' if args.before else 'stage2')/'bin/llama-server'
results=[]
for speculative in (False, True):
    results.append(namespace['run_case']('glm5next',speculative,out,server))
    namespace['save'](out/'summary.json',results)
raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)
