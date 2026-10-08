#!/usr/bin/env python3
import json
from pathlib import Path
R=Path(__file__).resolve().parent
c=json.loads((R/'real-comparison.json').read_text());out=['# Реальные модели: сравнение до/после','',f"Полных пар запусков: {c['complete_pairs']}/8; пар ответов: {c['paired_responses']}; совпали greedy: {c['greedy_equal']}/40, sampling: {c['sampling_equal']}/16.",'','| Модель / spec | Prompt tokens | Prefill до/после, токен/с | Decode до/после, токен/с | Prefill, изменение | Decode, изменение | Принято draft до/после |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
for p in c['pairs']:
    for m in p['performance']:
        b=m['before'];a=m['after'];n=b['samples'][0]['prompt_n'];acc='-'
        if b['acceptance_percent'] is not None:acc=f"{b['acceptance_percent']:.1f}% / {a['acceptance_percent']:.1f}%"
        out.append(f"| {p['case']} | {n} | {b['prompt_per_second']:.1f} / {a['prompt_per_second']:.1f} | {b['predicted_per_second']:.2f} / {a['predicted_per_second']:.2f} | {m['prefill_speed_change_percent']:+.2f}% | {m['decode_speed_change_percent']:+.2f}% | {acc} |")
out+=['','Для GLM5NEXT/Qwen указаны медианы трёх прогретых greedy запросов каждого prompt; для GLM-DSA/DeepSeek в основной серии после первого greedy остаётся один замер. Поэтому отдельное снижение GLM-DSA native проверяется дополнительной серией из шести greedy запросов после warmup.','',f"Совпадение параметров, слоёв, model buffers и boot ID: {sum(all(p[k] for k in ('parameters_equal','placement_equal','model_buffers_equal','boot_id_equal')) for p in c['pairs'])}/8.",'']
if (R/'dsa-validation-comparison.json').exists():
    v=json.loads((R/'dsa-validation-comparison.json').read_text());out+=['## Дополнительный GLM-DSA native контроль','',f"Warmup и шесть greedy запросов на каждой версии; оценка по медиане последних пяти. Совпали {v['equal_responses']}/7 пар ответов.",'',f"Prefill: {v['before']['prompt_per_second']:.2f} / {v['after']['prompt_per_second']:.2f} токен/с ({v['prefill_change_percent']:+.2f}%). Decode: {v['before']['predicted_per_second']:.2f} / {v['after']['predicted_per_second']:.2f} токен/с ({v['decode_change_percent']:+.2f}%).",'']
(R/'real-results.md').write_text('\n'.join(out)+'\n')
