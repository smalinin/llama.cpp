#!/usr/bin/env python3
"""Compare the same saved requests before and after a source change."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('before', type=Path)
parser.add_argument('after', type=Path)
args = parser.parse_args()
rows = []
for path in sorted((args.after / 'runs').glob('*/*-response.json')):
    relative = path.relative_to(args.after / 'runs')
    baseline = args.before / 'runs' / relative
    if not baseline.exists():
        continue
    before, after = (json.loads(p.read_text()) for p in (baseline, path))
    a, b = before['tokens'], after['tokens']
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    rows.append({
        'case': path.parent.name, 'request': path.name.removesuffix('-response.json'),
        'tokens_identical': a == b, 'content_identical': before['content'] == after['content'],
        'first_different_token': first, 'tokens_before': len(a), 'tokens_after': len(b),
        'stop_before': before['stop_type'], 'stop_after': after['stop_type'],
        'decode_tps_before': before['timings']['predicted_per_second'],
        'decode_tps_after': after['timings']['predicted_per_second'],
    })
(args.after / 'before-after-comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps({'compared': len(rows), 'different': [r for r in rows if not r['tokens_identical']]}, indent=2))
