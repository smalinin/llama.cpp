#!/usr/bin/env python3
"""Summarize saved baseline responses; no inference or source changes."""

import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    results = json.loads((root / 'runs/summary.json').read_text())
    metrics = []
    comparisons = []
    for result in results:
        directory = root / 'runs' / result['case']
        samples = [json.loads(line) for line in (directory / 'memory.jsonl').read_text().splitlines()]
        aggregate = max((sum(s['gpu_mib'].values()) for s in samples if len(s['gpu_mib']) == 6), default=0)
        for mode in ('greedy', 'sampling'):
            requests = [r for r in result['requests'] if r['request'].startswith(mode)]
            if not requests:
                continue
            timings = [r['timings'] for r in requests]
            drafted = sum(t.get('draft_n', 0) for t in timings)
            accepted = sum(t.get('draft_n_accepted', 0) for t in timings)
            metrics.append({
                'case': result['case'], 'status': result['status'], 'mode': mode,
                'requests': len(requests),
                'prefill_tps': [t['prompt_per_second'] for t in timings],
                'decode_tps': [t['predicted_per_second'] for t in timings],
                'mean_decode_tps': statistics.mean(t['predicted_per_second'] for t in timings),
                'drafted': drafted, 'accepted': accepted,
                'acceptance_pct': 100 * accepted / drafted if drafted else None,
                'repeat_identical': result.get(mode + '_repeat_identical'),
                'startup_s': result.get('startup_s'),
                'peak_process_rss_gib': result['peak_process_rss_kib'] / 1024**2,
                'peak_simultaneous_gpu_used_gib': aggregate / 1024,
                'peak_gpu_used_mib': result['peak_gpu_used_mib'],
            })
    models = dict.fromkeys(r['case'].rsplit('-spec-', 1)[0] for r in results)
    for model in models:
        for mode in ('greedy', 'sampling'):
            paths = [root / 'runs' / f'{model}-spec-{spec}' / f'{mode}-1-response.json' for spec in (0, 1)]
            if not all(p.exists() for p in paths):
                continue
            off, on = (json.loads(p.read_text()) for p in paths)
            a, b = off['tokens'], on['tokens']
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if first is None and len(a) != len(b):
                first = min(len(a), len(b))
            comparisons.append({
                'model': model, 'mode': mode,
                'content_identical': off['content'] == on['content'],
                'tokens_identical': a == b, 'first_different_token': first,
                'n_off': len(a), 'n_on': len(b),
            })
    for filename, data in [('baseline-metrics.json', metrics), ('off-on-comparison.json', comparisons)]:
        (root / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed_cases': len(results), 'statuses': {r['case']: r['status'] for r in results},
                      'comparisons': comparisons}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
