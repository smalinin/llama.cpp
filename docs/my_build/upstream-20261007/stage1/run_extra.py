#!/usr/bin/env python3
"""Additional direct-I/O and small-ubatch checks using the baseline runner."""
import json
from pathlib import Path

ROOT = Path('/home/sergei/_my_sync/llama_upstream_review')
SERVER = Path('/home/sergei/Github/llama.cpp/build-glm53/bin/llama-server')
source = (ROOT / 'run_baseline.py').read_text()
results = []
for label, model, speculative, changes in [
    ('direct-io', 'qwen4exp', False, [("'--load-mode', 'none'", "'--load-mode', 'dio'")]),
    ('small-ubatch', 'glm5next', False, [("'-ub', '512'", "'-ub', '16'")]),
    ('eos-reuse', 'deepseek41', True, []),
]:
    code = source
    for before, after in changes:
        assert before in code
        code = code.replace(before, after)
    if label == 'direct-io':
        hook = "                for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():"
        assert hook in code
        code = code.replace(hook, '''                direct_fds = 0
                for descriptor in Path(f'/proc/{proc.pid}/fdinfo').glob('*'):
                    try:
                        target = os.readlink(f'/proc/{proc.pid}/fd/{descriptor.name}')
                        flags = next(line.split()[1] for line in descriptor.read_text().splitlines() if line.startswith('flags:'))
                        if target.endswith('.gguf') and int(flags, 8) & os.O_DIRECT:
                            direct_fds += 1
                    except (OSError, StopIteration):
                        pass
                sample['gguf_direct_io_fds'] = direct_fds
''' + hook)
    if label == 'eos-reuse':
        hook = "            result['status'] = 'passed'"
        assert hook in code
        code = code.replace(hook, '''            reuse = []
            for repeat in (1, 2):
                request = json.loads((directory / 'greedy-1-request.json').read_text())
                request['cache_prompt'] = True
                response = api(port, '/completion', request)
                save(directory / f'reuse-{repeat}-request.json', request)
                save(directory / f'reuse-{repeat}-response.json', response)
                if 'error' in response:
                    raise RuntimeError(str(response['error']))
                reuse.append(response)
            original = json.loads((directory / 'greedy-1-response.json').read_text())
            result['eos_reuse'] = {
                'first_stop': original['stop_type'],
                'stop_types': [r['stop_type'] for r in reuse],
                'token_counts': [len(r['tokens']) for r in reuse],
                'cached_tokens': [r['timings'].get('cache_n', 0) for r in reuse],
                'repeat_identical': reuse[0]['tokens'] == reuse[1]['tokens'],
                'matches_cold': reuse[0]['tokens'] == original['tokens'],
            }
''' + hook)
    namespace = {'__name__': 'baseline_extra'}
    exec(compile(code, str(ROOT / 'run_baseline.py'), 'exec'), namespace)
    out = ROOT / 'stage1' / label
    out.mkdir(exist_ok=False)
    result = namespace['run_case'](model, speculative, out, SERVER)
    results.append({'check': label, **result})
(ROOT / 'stage1/extra-summary.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(0 if all(r['status'] == 'passed' for r in results) else 1)
