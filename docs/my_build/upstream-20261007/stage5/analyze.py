#!/usr/bin/env python3
"""Summarize independent numerical, capture and server evidence."""
from array import array
from collections import Counter
import hashlib
import ctypes
import json
import math
from pathlib import Path
import re
import statistics

R = Path(__file__).resolve().parent
native = None
if (R / 'metrics.so').exists():
    native = ctypes.CDLL(str(R / 'metrics.so')).stage5_metrics
    native.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    native.restype = None


def save(name, data):
    (R / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def metrics(a, b, masked=False):
    if native:
        assert len(a) == len(b) and len(a) % 4 == 0
        values = (ctypes.c_double * 5)()
        native(a, b, len(a)//4, masked, values)
        return {'bitwise_equal': a == b, 'nmse': values[0]/max(values[1], 1e-30), 'max_abs': values[2],
                'masked_changes': int(values[3]), 'finite': values[4] == 0}
    if a == b:
        values = array('f')
        values.frombytes(a)
        return {'bitwise_equal': True, 'nmse': 0, 'max_abs': 0, 'finite': all(math.isfinite(x) for x in values)}
    x, y = array('f'), array('f')
    x.frombytes(a)
    y.frombytes(b)
    assert len(x) == len(y)
    err = norm = maxabs = 0.0
    finite = True
    mask_changes = 0
    for p, q in zip(x, y):
        if not math.isfinite(p) or not math.isfinite(q):
            finite = False
            continue
        if masked and (p < -1e8 or q < -1e8):
            mask_changes += p != q
            continue
        d = p - q
        err += d*d
        norm += p*p
        maxabs = max(maxabs, abs(d))
    return {'bitwise_equal': False, 'nmse': err/max(norm, 1e-30), 'max_abs': maxabs,
            'finite': finite, 'masked_changes': mask_changes}


checks = {}
for gpu in ('4', '3'):
    p = R / ('qsa-derived-mask-gpu' + gpu + '.log')
    if not p.exists():
        p = R / ('qsa-numerical-gpu' + gpu + '.log')
    if p.exists():
        cases = [json.loads(line) for line in p.read_text().splitlines() if line.startswith('{')]
        checks[gpu] = {'cases': len(cases), 'failed': sum(not c['passed'] for c in cases),
                       'max_nmse': max((c['nmse'] for c in cases), default=0),
                       'ties': sum(c['tie'] for c in cases),
                       'tie_selection_changes': sum(c['selection_changes'] for c in cases if c['tie']),
                       'ordinary_order_changes': sum(c['selection_changes'] for c in cases if not c['tie']),
                       'ordinary_set_changes': sum(c.get('selection_sets_changed', 0) for c in cases if not c['tie'])}
save('numerical-summary.json', checks)
p = R / 'qsa-derived-memory-reserve.log'
if not p.exists():
    p = R / 'qsa-memory-reserve.log'
if p.exists():
    save('memory-reserve.json', [json.loads(s) for s in p.read_text().splitlines() if s.startswith('{')])

for versions, capture_name in [(('before', 'after'), 'capture-comparison.json'), (('before', 'final'), 'capture-default-comparison.json'), (('after', 'final-optin'), 'capture-optin-comparison.json'), (('before', 'review-default'), 'capture-review-default-comparison.json'), (('after', 'review'), 'capture-review-optin-comparison.json'), (('before', 'candidate-default'), 'capture-candidate-default-comparison.json'), (('review', 'candidate'), 'capture-candidate-optin-comparison.json')]:
    captures = []
    completed = set()
    for status in ('verification-progress.json', 'final-progress.json', 'review-progress.json', 'candidate-progress.json'):
        if (R / status).exists():
            completed.update(c['case'] for c in json.loads((R / status).read_text()) if c['exit_code'] == 0)
    for slots in (1, 2):
        dirs = [R / f'capture-{v}-np{slots}' for v in versions]
        if not all(f'capture-{v}-np{slots}' in completed for v in versions):
            continue
        before, after = [[json.loads(s) for s in (d / 'tensors.jsonl').read_text().splitlines()] for d in dirs]
        raw_counts = [len(before), len(after)]
        exact_name = re.compile(r'^indexer_(?:q|k|score_blk|top_k_blk)-\d+$')
        maps = []
        for events in (before, after):
            index = {}
            for event in events:
                if not exact_name.fullmatch(event['name']):
                    continue
                key = event['step'], event['name']
                assert key not in index
                index[key] = event
            maps.append(index)
        assert maps[0] and maps[0].keys() == maps[1].keys()
        keys = sorted(maps[0])
        before, after = [[m[k] for k in keys] for m in maps]
        groups = {}
        records = []
        for a, b in zip(before, after):
            assert (a['name'], a['step'], a['ne'], a['type']) == (b['name'], b['step'], b['ne'], b['type'])
            x, y = (dirs[0] / a['file']).read_bytes(), (dirs[1] / b['file']).read_bytes()
            if a['type'] == 26:
                m = {'bitwise_equal': x == y}
                if x != y:
                    ix, iy = array('i'), array('i')
                    ix.frombytes(x)
                    iy.frombytes(y)
                    m['positions_changed'] = sum(p != q for p, q in zip(ix, iy))
                    row = a['ne'][0]
                    m['sets_changed'] = sum(set(ix[i:i+row]) != set(iy[i:i+row]) for i in range(0, len(ix), row))
            else:
                m = metrics(x, y, 'score_blk' in a['name'])
            label = a['name'].rsplit('-', 1)[0]
            g = groups.setdefault(label, {'tensors': 0, 'bitwise_equal': 0, 'max_nmse': 0, 'max_abs': 0,
                                          'positions_changed': 0, 'sets_changed': 0, 'finite': True, 'masked_changes': 0})
            g['tensors'] += 1
            g['bitwise_equal'] += m['bitwise_equal']
            for key in ('max_nmse', 'max_abs'):
                g[key] = max(g[key], m.get(key.replace('max_nmse', 'nmse'), 0))
            for key in ('positions_changed', 'sets_changed', 'masked_changes'):
                g[key] += m.get(key, 0)
            g['finite'] = g['finite'] and m.get('finite', True)
            if not m['bitwise_equal']:
                records.append({'name': a['name'], 'step': a['step'], 'ne': a['ne'], **m})
        libs = []
        for version, d in zip(versions, dirs):
            paths = sorted({s.split()[-1] for s in (d / 'loaded-libraries.txt').read_text().splitlines()})
            assert paths and all(Path(s).parent.resolve() == (R / (({'final-optin': 'final', 'review-default': 'review', 'candidate-default': 'candidate'}.get(version, version)) + '-bin')).resolve() for s in paths)
            libs.append({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
        captures.append({'slots': slots, 'tensors': len(before), 'raw_event_counts': raw_counts, 'groups': groups,
                         'greedy_equal': (dirs[0] / 'generated.jsonl').read_bytes() == (dirs[1] / 'generated.jsonl').read_bytes(),
                         'logits': metrics((dirs[0] / 'logits.bin').read_bytes(), (dirs[1] / 'logits.bin').read_bytes()),
                         'loaded_libraries': libs, 'changed_tensors': records})
    save(capture_name, captures)

for version in ('after', 'final', 'review', 'candidate'):
    pairs = []
    for a in sorted((R / 'real-before').glob('*/result.json')):
        b = R / ('real-' + version) / a.parent.name / 'result.json'
        if not b.exists():
            continue
        before, after = json.loads(a.read_text()), json.loads(b.read_text())
        if after['status'] != 'passed':
            continue
        ma, mb = [json.loads((p.parent / 'manifest.json').read_text()) for p in (a, b)]
        row = {'case': a.parent.name, 'conditions_equal': all(ma[k] == mb[k] for k in
               ('gpu_order', 'boot_id', 'context', 'slots', 'mtp', 'layer_placement', 'model_buffers')), 'responses': []}
        def normalize(command):
            command = command[1:]
            command[command.index('--port')+1] = '<port>'
            return command
        row['commands_equal'] = normalize(ma['command']) == normalize(mb['command'])
        def buffers(manifest):
            result = {}
            for line in manifest['compute_buffers']:
                m = re.search(r'(CUDA\d+|CUDA_Host|CPU) compute buffer size =\s*([\d.]+) MiB', line)
                if m:
                    result[m[1]] = float(m[2])
            return result
        row['compute_before_mib'], row['compute_after_mib'] = buffers(ma), buffers(mb)
        row['peak_gpu_before_mib'], row['peak_gpu_after_mib'] = before['peak_gpu_used_mib'], after['peak_gpu_used_mib']
        row['peak_rss_before_kib'], row['peak_rss_after_kib'] = before['peak_process_rss_kib'], after['peak_process_rss_kib']
        for p in sorted(a.parent.glob('*-response.json')):
            q = b.parent / p.name
            x, y = json.loads(p.read_text()), json.loads(q.read_text())
            row['responses'].append({'request': p.stem, 'tokens_equal': x['tokens'] == y['tokens'],
                                     'text_equal': x['content'] == y['content'],
                                     'draft_counts_equal': all(x['timings'].get(k, 0) == y['timings'].get(k, 0)
                                                               for k in ('draft_n', 'draft_n_accepted')),
                                     'before_timings': x['timings'], 'after_timings': y['timings']})
        row['performance'] = {}
        for phase in ('short', 'long'):
            selected = [x for x in row['responses'] if x['request'].startswith(phase+'-2-')]
            if selected:
                row['performance'][phase] = {}
                for key in ('prompt_per_second', 'predicted_per_second'):
                    x, y = [statistics.median(r[v+'_timings'][key] for r in selected) for v in ('before', 'after')]
                    row['performance'][phase][key] = {'before': x, 'after': y, 'change_percent': 100*(y/x-1)}
        pairs.append(row)
    save('real-comparison.json' if version == 'after' else 'final-default-comparison.json' if version == 'final' else 'review-optin-comparison.json' if version == 'review' else 'candidate-optin-comparison.json', {'pairs': pairs, 'paired_responses': sum(len(p['responses']) for p in pairs),
                                  'unequal_responses': [dict(case=p['case'], **r) for p in pairs for r in p['responses']
                                                        if not r['tokens_equal'] or not r['text_equal']],
                                  'draft_counter_mismatches': sum(not r['draft_counts_equal'] for p in pairs for r in p['responses'])})
print('Numerical GPU cases:', checks)
print('Capture comparisons updated.')
print('Model comparisons updated.')
