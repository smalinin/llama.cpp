#!/usr/bin/env python3
"""Compare saved token IDs and text within and across server launches."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(itertools.zip_longest(a, b)) if x != y), None)


def compare(a, b):
    left, right = a['data'], b['data']
    index = first_diff(left['tokens'], right['tokens'])
    return {'a': a['id'], 'b': b['id'], 'tokens_identical': index is None,
            'text_identical': left['content'] == right['content'],
            'first_different_index_0based': index,
            'first_different_token_1based': None if index is None else index + 1,
            'a_token': left['tokens'][index] if index is not None and index < len(left['tokens']) else None,
            'b_token': right['tokens'][index] if index is not None and index < len(right['tokens']) else None}


def group(items):
    comparisons = [compare(a, b) for a, b in itertools.combinations(items, 2)]
    differences = [c['first_different_index_0based'] for c in comparisons if not c['tokens_identical']]
    drafts = sum(item['data']['timings'].get('draft_n', 0) for item in items)
    accepted = sum(item['data']['timings'].get('draft_n_accepted', 0) for item in items)
    return {'requests': len(items), 'all_token_ids_identical': not differences,
            'all_text_identical': all(c['text_identical'] for c in comparisons),
            'unique_token_sequences': len({tuple(item['data']['tokens']) for item in items}),
            'unique_texts': len({item['data']['content'] for item in items}),
            'earliest_different_index_0based': min(differences, default=None),
            'drafted_total': drafts, 'accepted_total': accepted,
            'acceptance_ratio': accepted / drafts if drafts else None,
            'unique_draft_counts': sorted({(item['data']['timings'].get('draft_n', 0),
                                            item['data']['timings'].get('draft_n_accepted', 0))
                                           for item in items}),
            'comparisons': comparisons}


def analyze(runs, baseline):
    summary = {'tested_source_head': '11638b68545860e96b055798e995bb14be3d0e88',
               'includes_pending_continue_fix': False, 'groups': {}, 'within_launch': {},
               'cross_launch': {}, 'requests': [], 'placement': {}, 'model_files_match_baseline': True,
               'embedded_build_info': {}}
    groups = {}
    manifests = {}
    baseline_files = read(baseline / 'model-files.json')
    expected = {item['path']: item for item in baseline_files if 'GLM-5.3-UD-IQ3_XXS-' in item['path']}
    for mtp in (0, 1):
        for launch in (1, 2):
            case = f'mtp{mtp}-launch{launch}'
            directory = runs / case
            result = read(directory / 'result.json')
            if result['status'] != 'passed' or result['server_exit_code'] != 0:
                raise RuntimeError('incomplete run: ' + case)
            manifest = read(directory / 'manifest.json')
            manifests[case] = manifest
            summary['embedded_build_info'][case] = read(directory / 'props.json')['build_info']
            if not manifest.get('layer_placement') or not manifest.get('loaded_libraries'):
                raise RuntimeError('missing provenance: ' + case)
            for item in manifest['model_files']:
                old = expected.get(item['path'])
                if old is None or old['bytes'] != item['size'] or old['mtime_ns'] != item['mtime_ns']:
                    summary['model_files_match_baseline'] = False
            for mode in (['sampling', 'greedy'] if mtp else ['sampling']):
                items = []
                for repeat in range(1, 6):
                    name = f'{mode}-{repeat}'
                    response = read(directory / f'{name}-response.json')
                    payload = read(directory / f'{name}-request.json')
                    original = read(baseline / f'runs/glm-dsa-spec-{mtp}/{mode}-1-request.json')
                    if payload != original:
                        raise RuntimeError('request differs from baseline: ' + case + '/' + name)
                    if len(response['tokens']) != 128 or response['timings']['prompt_n'] != 1656:
                        raise RuntimeError('unexpected token count: ' + case + '/' + name)
                    item = {'id': case + '/' + name, 'launch': launch, 'data': response}
                    items.append(item)
                    summary['requests'].append({'id': item['id'], 'timings': response['timings'],
                        'tokens_sha256': hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest(),
                        'text_sha256': hashlib.sha256(response['content'].encode()).hexdigest()})
                key = f'mtp{mtp}-{mode}'
                groups.setdefault(key, []).extend(items)
                summary['within_launch'][case + '-' + mode] = group(items)
    for key, items in groups.items():
        summary['groups'][key] = group(items)
        first = [item for item in items if item['launch'] == 1]
        second = [item for item in items if item['launch'] == 2]
        pairs = [compare(a, b) for a in first for b in second]
        summary['cross_launch'][key] = {'comparisons': pairs,
            'all_token_ids_identical': all(p['tokens_identical'] for p in pairs),
            'all_text_identical': all(p['text_identical'] for p in pairs)}
    for mtp in (0, 1):
        a, b = (manifests[f'mtp{mtp}-launch{i}'] for i in (1, 2))
        summary['placement'][f'mtp{mtp}-repeat_identical'] = a['layer_placement'] == b['layer_placement']
    summary['placement']['off_on_identical'] = manifests['mtp0-launch1']['layer_placement'] == manifests['mtp1-launch1']['layer_placement']
    summary['placement']['note'] = 'Automatic fit can change placement when the draft context is enabled.'
    old_a = {'id': 'stage0/sampling-1', 'data': read(baseline / 'runs/glm-dsa-spec-1/sampling-1-response.json')}
    old_b = {'id': 'stage0/sampling-2', 'data': read(baseline / 'runs/glm-dsa-spec-1/sampling-2-response.json')}
    summary['original_baseline_mtp_sampling'] = compare(old_a, old_b)
    summary['request_payloads_match_baseline'] = True
    summary['total_requests'] = len(summary['requests'])
    summary['all_cache_n_zero'] = all(item['timings'].get('cache_n', 0) == 0 for item in summary['requests'])
    summary['all_modes_repeatable'] = all(item['all_token_ids_identical'] and item['all_text_identical']
                                          for item in summary['groups'].values())
    summary['pairs_compared'] = sum(len(item['comparisons']) for item in summary['groups'].values())
    summary['all_loaded_libraries_equal'] = len({json.dumps(item['loaded_libraries'], sort_keys=True)
                                                for item in manifests.values()}) == 1
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--baseline', type=Path, default=ROOT.parent / 'stage0')
    args = parser.parse_args()
    summary = analyze(args.runs, args.baseline)
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for name, result in summary['groups'].items():
        print(name, json.dumps({k: v for k, v in result.items() if k != 'comparisons'}))
    print('placement', summary['placement'])


if __name__ == '__main__':
    main()
