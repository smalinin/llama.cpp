#!/usr/bin/env python3
"""Check a running Step-5 server with mixed archives and cached follow-ups."""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

FACTS = dict(zip(('ALPHA', 'BETA', 'GAMMA', 'DELTA', 'EPSILON', 'ZETA', 'ETA', 'THETA'),
                 ('865421', '390217', '704193', '258640', '619835', '472906', '183752', '936048')))


def archive(units):
    positions = {int((units - 1) * j / 7): pair for j, pair in enumerate(FACTS.items())}
    parts = ['<|im_start|>user\nRead this mixed archive. Keep the verification code associated with each named RECORD. Other document numbers are ordinary background data.\n']
    for i in range(units):
        if i in positions:
            name, value = positions[i]
            parts.append(f'\nRECORD {name}: the verification number is {value}.\n')
        variant = i % 6
        if variant == 0:
            text = f'\u0416\u0443\u0440\u043d\u0430\u043b \u0441\u0442\u0430\u043d\u0446\u0438\u0438 {i:05d}. \u0423\u0442\u0440\u043e\u043c \u0441\u043e\u0442\u0440\u0443\u0434\u043d\u0438\u043a\u0438 \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b\u0438 \u0434\u043e\u0440\u043e\u0433\u0438, \u043c\u043e\u0441\u0442 \u0438 \u0441\u043a\u043b\u0430\u0434. \u0411\u0443\u043c\u0430\u0436\u043d\u044b\u0435 \u043e\u0442\u0447\u0451\u0442\u044b \u0441\u0432\u0435\u0440\u0435\u043d\u044b \u0441 \u0436\u0443\u0440\u043d\u0430\u043b\u043e\u043c; \u043d\u043e\u0432\u044b\u0445 \u0440\u0430\u0441\u043f\u043e\u0440\u044f\u0436\u0435\u043d\u0438\u0439 \u043d\u0435\u0442.'
        elif variant == 1:
            text = f'Python utility {i:05d}:\ndef normalize_{i%97}(items):\n    return [str(item).strip().lower() for item in items if item is not None]\n# The utility only changes spelling and removes empty records.'
        elif variant == 2:
            text = json.dumps({'document': f'D{i:05d}', 'status': 'reviewed', 'route': f'R{i%83}', 'items': i % 47 + 1, 'notes': ['paper archive', 'routine shipment']})
        elif variant == 3:
            text = f'| office | boxes | condition |\n| East-{i%31} | {i%29+2} | dry |\n| West-{i%17} | {i%19+3} | sealed |\nThe table is an inventory, with no verification codes.'
        elif variant == 4:
            text = f'GET /v1/archive/{i:05d} status=200 duration={i%89+3}ms\nThe request used a read-only transaction. The response contained the normal catalog header and an empty list of alerts.'
        else:
            text = f'Document {i:05d}: The archive contains notes on weather, roads, trees, maintenance and paper. A clerk checked the local index, closed the ledger and returned it to the northern shelf.'
        parts.append(f'\n--- DOCUMENT {i:05d} ---\n{text}\n')
    parts.append('\nQuestion: Give the verification number for each of ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA, ETA and THETA. Reply only as NAME=number, one per line.\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n')
    return ''.join(parts)


def check_pairs(content, expected):
    found = {name: re.findall(r'\b' + name + r'\b\s*[=:]\s*(\d+)', content) for name in expected}
    return found, all(values and set(values) == {expected[name]} for name, values in found.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8086')
    p.add_argument('--tokens', type=int, nargs='+', default=[7000, 15000, 31000, 60000, 143000])
    p.add_argument('--request', type=Path, help='Use a saved eight-fact archive completion request')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--follow-up', action='store_true')
    p.add_argument('--repeat', type=int, default=1)
    a = p.parse_args()
    if a.repeat < 1 or any(n < 512 for n in a.tokens):
        p.error('repeat must be positive and target lengths must be at least 512')
    a.output.mkdir(parents=True, exist_ok=True)

    def post(endpoint, body):
        request = urllib.request.Request(a.url.rstrip('/') + endpoint, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=7200) as response:
            return json.load(response)

    def count(prompt):
        return len(post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens'])

    records = []
    for target in ([None] if a.request else a.tokens):
        if a.request:
            body = json.loads(a.request.read_text())
            name = a.request.stem
        else:
            lo, hi = 8, max(16, target // 40)
            while count(archive(hi)) < target:
                hi *= 2
            while hi - lo > 1:
                middle = (lo + hi) // 2
                if count(archive(middle)) <= target:
                    lo = middle
                else:
                    hi = middle
            body = {'prompt': archive(lo), 'n_predict': 192, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'return_tokens': True}
            name = f'mixed-{target}'
        (a.output / (name + '-request.json')).write_text(json.dumps(body, ensure_ascii=False, indent=2) + '\n')
        for repetition in range(a.repeat):
            started = time.monotonic()
            result = post('/completion', body)
            result['_wall_seconds'] = time.monotonic() - started
            (a.output / f'{name}-{repetition}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            found, passed = check_pairs(result.get('content', ''), FACTS)
            passed = passed and not result.get('truncated', False) and result.get('stop_type') != 'limit'
            record = {'name': name, 'repetition': repetition, 'prompt_sha256': hashlib.sha256(body['prompt'].encode()).hexdigest(), 'found': found, 'passed': passed, 'timings': result.get('timings'), 'stop_type': result.get('stop_type'), 'wall_seconds': result['_wall_seconds']}
            records.append(record)
            print(json.dumps(record), flush=True)
            if a.follow_up:
                cached = dict(body, cache_prompt=True)
                repeat_result = post('/completion', cached)
                follow_prompt = body['prompt'] + result['content'] + '<|im_end|>\n<|im_start|>user\nGive THETA and ALPHA again, in that order, as NAME=number.\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n'
                follow_result = post('/completion', dict(body, prompt=follow_prompt, cache_prompt=True, n_predict=64))
                for suffix, response, expected in [('cached', repeat_result, FACTS), ('follow-up', follow_result, {k: FACTS[k] for k in ('THETA', 'ALPHA')})]:
                    (a.output / f'{name}-{repetition}-{suffix}.json').write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n')
                    found, passed = check_pairs(response.get('content', ''), expected)
                    passed = passed and not response.get('truncated', False) and response.get('stop_type') != 'limit'
                    item = {'name': name + '-' + suffix, 'repetition': repetition, 'found': found, 'passed': passed, 'timings': response.get('timings'), 'stop_type': response.get('stop_type')}
                    records.append(item)
                    print(json.dumps(item), flush=True)
            (a.output / 'report.json').write_text(json.dumps(records, indent=2) + '\n')
    if not all(r['passed'] for r in records):
        raise SystemExit('Fact retrieval failed; see saved responses')


if __name__ == '__main__':
    main()
