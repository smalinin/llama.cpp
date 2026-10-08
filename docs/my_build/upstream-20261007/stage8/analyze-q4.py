#!/usr/bin/env python3
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / 'q4-comparison/results.json').read_text())
assert len(results) == 2 and all(r['status'] == 'passed' and r['server_exit_code'] == 0 for r in results)
summary = {'config': 'n3-p0', 'repeats': 3, 'binaries': 'Stage5 snapshot; pending FP32 fix excluded',
           'models': {}, 'prompts': {}}
for result in results:
    label = result['draft_variant']
    directory = ROOT / 'q4-comparison' / label / 'n3-p0'
    samples = [json.loads(line) for line in (directory / 'memory.jsonl').read_text().splitlines()]
    summary['models'][label] = {'requests': len(result['requests']), 'exit_code': result['server_exit_code'],
                               'peak_gpu_sum_concurrent_mib': max(sum(s['gpu_mib'].values()) for s in samples),
                               'peak_process_rss_kib': result['peak_process_rss_kib']}
    for name in ('baseline', 'svg', 'python', 'explain'):
        items = [x for x in result['requests'] if x['prompt'] == name and x['phase'] == 'measure']
        assert len(items) == 3
        responses = [json.loads((directory / (x['request'] + '-response.json')).read_text()) for x in items]
        assert all(x['tokens'] == responses[0]['tokens'] and x['content'] == responses[0]['content'] for x in responses)
        assert all(x['timings']['cache_n'] == 0 for x in responses)
        data = {'decode_tps': [x['timings']['predicted_per_second'] for x in items],
                'decode_tps_median': statistics.median(x['timings']['predicted_per_second'] for x in items),
                'wall_s_median': statistics.median(x['wall_s'] for x in items),
                'prompt_ms_median': statistics.median(x['timings']['prompt_ms'] for x in items),
                'output_length': len(responses[0]['tokens']), 'stable_tokens_and_text': True,
                'draft_n': responses[0]['timings']['draft_n'],
                'draft_n_accepted': responses[0]['timings']['draft_n_accepted']}
        data['acceptance'] = data['draft_n_accepted'] / data['draft_n']
        summary['prompts'].setdefault(name, {})[label] = data
for name, data in summary['prompts'].items():
    files = {label: ROOT / 'q4-comparison' / label / 'n3-p0' / (name + '-measure-1-response.json')
             for label in ('mxfp4', 'q4k')}
    a, b = (json.loads(files[label].read_text()) for label in ('mxfp4', 'q4k'))
    data['same_tokens_between_formats'] = a['tokens'] == b['tokens']
    data['same_text_between_formats'] = a['content'] == b['content']
    data['first_token_difference'] = next((i for i, pair in enumerate(zip(a['tokens'], b['tokens'])) if pair[0] != pair[1]),
                                         min(len(a['tokens']), len(b['tokens'])) if len(a['tokens']) != len(b['tokens']) else None)
    data['q4k_rate_change_percent'] = 100 * (data['q4k']['decode_tps_median'] / data['mxfp4']['decode_tps_median'] - 1)
    print(name, 'MXFP4', round(data['mxfp4']['decode_tps_median'], 2),
          'Q4_K', round(data['q4k']['decode_tps_median'], 2),
          'delta %', round(data['q4k_rate_change_percent'], 2),
          'same tokens', data['same_tokens_between_formats'])
(ROOT / 'q4-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
