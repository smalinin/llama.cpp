#!/usr/bin/env python3
"""Summarize DSpark measurements and target-only teacher forcing."""
import argparse
from array import array
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def first_difference(left, right):
    return next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                min(len(left), len(right)) if len(left) != len(right) else None)

def benchmark(directory):
    summary = {'measurement_rounds':3, 'n_predict':256, 'diagnostic_n_predict':128,
               'warming_excluded':True, 'configs':{}}
    for path in sorted(directory.glob('*/result.json')):
        result = read(path)
        if result['status'] != 'passed':
            continue
        config = result['config']
        entry = {'status':result['status'], 'request_count':len(result['requests']),
                 'peak_gpu_used_mib':result['peak_gpu_used_mib'],
                 'peak_process_rss_kib':result['peak_process_rss_kib'], 'prompts':{}}
        samples = [json.loads(line) for line in (path.parent/'memory.jsonl').read_text().splitlines()]
        entry['peak_gpu_sum_concurrent_mib'] = max(sum(sample['gpu_mib'].values()) for sample in samples)
        for prompt in ('baseline', 'svg', 'python', 'explain'):
            items = [item for item in result['requests'] if item['prompt']==prompt and item['phase']=='measure']
            if not items:
                continue
            responses = [read(path.parent/(item['request']+'-response.json')) for item in items]
            native = read(directory/'off'/(items[0]['request']+'-response.json'))
            rates = [item['timings']['predicted_per_second'] for item in items]
            draft = sum(item['timings'].get('draft_n',0) for item in items)
            accepted = sum(item['timings'].get('draft_n_accepted',0) for item in items)
            entry['prompts'][prompt] = {
                'decode_tps_median':statistics.median(rates), 'decode_tps_min':min(rates),
                'decode_tps_max':max(rates), 'wall_s_median':statistics.median(item['wall_s'] for item in items),
                'prompt_ms_median':statistics.median(item['timings']['prompt_ms'] for item in items),
                'output_lengths':[len(response['tokens']) for response in responses],
                'native_output_length':len(native['tokens']),
                'same_as_native':[response['tokens']==native['tokens'] for response in responses],
                'first_difference_native_zero_based':[first_difference(native['tokens'],response['tokens']) for response in responses],
                'stable_token_ids':all(response['tokens']==responses[0]['tokens'] for response in responses),
                'stable_text':all(response['content']==responses[0]['content'] for response in responses),
                'draft_n':draft, 'draft_accepted':accepted,
                'acceptance':accepted/draft if draft else None,
                'stop_types':[response['stop_type'] for response in responses],
            }
        native = read(directory/'off/baseline-diagnostic-1-response.json')
        actual = read(path.parent/'baseline-diagnostic-1-response.json')
        other = read(path.parent/'baseline-diagnostic-2-response.json')
        index = first_difference(native['tokens'],actual['tokens'])
        entry['diagnostic'] = {'first_difference_native_zero_based':index,
                               'native_length':len(native['tokens']), 'actual_length':len(actual['tokens']),
                               'stable':actual['tokens']==other['tokens'],
                               'native_at_difference':native['completion_probabilities'][index] if index is not None and index<len(native['tokens']) else None,
                               'actual_at_difference':actual['completion_probabilities'][index] if index is not None and index<len(actual['tokens']) else None}
        summary['configs'][config] = entry
    native = summary['configs'].get('off',{}).get('prompts',{})
    for entry in summary['configs'].values():
        for name, data in entry['prompts'].items():
            data['rate_ratio_vs_native'] = data['decode_tps_median']/native[name]['decode_tps_median']
    save(directory.parent/'benchmark-summary.json',summary)
    for config,entry in summary['configs'].items():
        print(config,[(name,round(data['decode_tps_median'],2),round(data['rate_ratio_vs_native'],3),
                       data['output_lengths'][0],data['first_difference_native_zero_based'][0]) for name,data in entry['prompts'].items()])

def profile(directory):
    summary = {}
    for path in directory.glob('*/result.json'):
        result = read(path)
        raw = (path.parent/'server.log').read_bytes()
        requests = []
        for request in result['requests']:
            part = raw[request['log_start_offset']:request['log_end_offset']].decode(errors='replace')
            stages = {}
            events = []
            for line in part.splitlines():
                if 'DSPARK_PROFILE stage=' not in line:
                    continue
                fields = dict(re.findall(r'(\w+)=([^\s]+)',line))
                stage = fields['stage']
                parsed = {}
                for key,value in fields.items():
                    if key == 'stage': parsed[key] = value
                    elif key == 'confidence': parsed[key] = [float(x) for x in value.split(',')]
                    else:
                        try: parsed[key] = int(value)
                        except ValueError: parsed[key] = float(value)
                events.append(parsed)
                stat = stages.setdefault(stage,{'count':0,'time_us_sum':0,'times_us':[]})
                stat['count'] += 1
                elapsed = parsed.get('time_us',0)
                stat['time_us_sum'] += elapsed
                if 'time_us' in parsed: stat['times_us'].append(elapsed)
                if stage == 'select':
                    stat['confidence_us_sum'] = stat.get('confidence_us_sum',0)+parsed['confidence_us']
                    stat['select_us_sum'] = stat.get('select_us_sum',0)+parsed['select_us']
                if stage == 'target_sampling':
                    for key in ('draft_k','accepted','rollback'):
                        stat[key+'_sum'] = stat.get(key+'_sum',0)+parsed[key]
                if stage in ('target_verify','target_decode'):
                    stat['width_counts'] = stat.get('width_counts',{})
                    width = str(parsed['verify_width'])
                    stat['width_counts'][width] = stat['width_counts'].get(width,0)+1
            for stat in stages.values():
                values = stat.pop('times_us')
                if values: stat.update(time_us_median=statistics.median(values),time_us_max=max(values))
            label = request['request']
            save(path.parent/(label+'-profile-events.json'),events)
            requests.append({'request':label,'timings':request['timings'],'wall_s':request['wall_s'],'stages':stages})
        summary[result['config']] = requests
    save(directory.parent/(directory.name+'-summary.json'),summary)
    print(json.dumps(summary,indent=2))

def replay(directory):
    files = sorted(directory.glob('*-rows.jsonl'))
    rows = {path.name.removesuffix('-rows.jsonl'):[json.loads(line) for line in path.read_text().splitlines()] for path in files}
    logits = {}
    for label in rows:
        values = array('f')
        with (directory/(label+'-logits.f32')).open('rb') as source:
            values.fromfile(source,(directory/(label+'-logits.f32')).stat().st_size//4)
        assert len(values)==65*129280
        logits[label] = values
    reference = rows['plain-w1']
    base = logits['plain-w1']
    summary = {'vocab':129280,'rows':65,'native_server_reference_matches_plain_w1':
               all(row['argmax']==row['reference'] for row in reference),'variants':{}}
    for label, actual in rows.items():
        differences = [i for i,(left,right) in enumerate(zip(reference,actual)) if left['argmax']!=right['argmax']]
        max_abs = 0.0
        sqsum = 0.0
        for left,right in zip(base,logits[label]):
            diff = abs(left-right)
            max_abs = max(max_abs,diff)
            sqsum += diff*diff
        idx = 33
        offset = idx*129280
        summary['variants'][label] = {
            'argmax_difference_indices_vs_plain_w1':differences,
            'reference_mismatch_indices':[i for i,row in enumerate(actual) if row['argmax']!=row['reference']],
            'max_abs_logits_vs_plain_w1':max_abs,'rms_logits_vs_plain_w1':math.sqrt(sqsum/len(base)),
            'index33':actual[idx],
            'index33_entries_minus_checkpoint':logits[label][offset+23914]-logits[label][offset+72888],
        }
    save(directory/'summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('benchmark','profile','replay'))
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    {'benchmark':benchmark,'profile':profile,'replay':replay}[args.mode](args.directory)
