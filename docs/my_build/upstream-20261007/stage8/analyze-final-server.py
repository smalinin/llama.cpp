#!/usr/bin/env python3
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
directory = ROOT / 'final-server-runs'
results = json.loads((directory / 'results.json').read_text())
assert len(results) == 2
summary = {'scope': 'One baseline prompt, three greedy completions per process; first request excluded from timing',
           'configurations': {}}
responses = {}
for result in results:
    assert result['status'] == 'passed' and result['server_exit_code'] == 0
    config = result['config']
    items = [json.loads((directory / config / (r['request'] + '-response.json')).read_text()) for r in result['requests']]
    assert len(items) == 3
    assert all(x['tokens'] == items[0]['tokens'] and x['content'] == items[0]['content'] for x in items)
    assert all(x['timings']['cache_n'] == 0 for x in items)
    responses[config] = items[0]
    summary['configurations'][config] = {
        'stable_tokens_and_text': True, 'output_length': len(items[0]['tokens']),
        'stop_type': items[0]['stop_type'],
        'decode_tps_median_last_two': statistics.median(x['timings']['predicted_per_second'] for x in items[1:]),
        'timings': [x['timings'] for x in items]}
a, b = responses['off'], responses['n3-p0']
summary['same_tokens_off_on'] = a['tokens'] == b['tokens']
summary['same_text_off_on'] = a['content'] == b['content']
summary['first_difference_zero_based'] = next((i for i, pair in enumerate(zip(a['tokens'], b['tokens'])) if pair[0] != pair[1]),
                                               min(len(a['tokens']), len(b['tokens'])) if len(a['tokens']) != len(b['tokens']) else None)
summary['speed_change_percent'] = 100*(summary['configurations']['n3-p0']['decode_tps_median_last_two'] /
                                      summary['configurations']['off']['decode_tps_median_last_two'] - 1)
(ROOT / 'final-greedy-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
