#!/usr/bin/env python3
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
before=R/'validation-before/glm-dsa-spec-0';after=R/'validation-after/glm-dsa-spec-0'
rb=json.loads((before/'result.json').read_text());ra=json.loads((after/'result.json').read_text())
if rb['status']!='passed' or ra['status']!='passed':raise SystemExit('repeat control not finished')
mb=json.loads((before/'manifest.json').read_text());ma=json.loads((after/'manifest.json').read_text())
responses=[]
for name in ['warmup']+[f'greedy-{i}' for i in range(1,7)]:
    b=json.loads((before/(name+'-response.json')).read_text());a=json.loads((after/(name+'-response.json')).read_text())
    responses.append({'request':name,'tokens_equal':b['tokens']==a['tokens'],'text_equal':b['content']==a['content'],'before':b['timings'],'after':a['timings']})
metrics={}
for version,path in [('before',before),('after',after)]:
    timing=[json.loads((path/f'greedy-{i}-response.json').read_text())['timings'] for i in range(2,7)]
    metrics[version]={k:statistics.median(t[k] for t in timing) for k in ('prompt_per_second','predicted_per_second')}
    metrics[version]['samples']=timing
    metrics[version]['decode_range']=[min(t['predicted_per_second'] for t in timing),max(t['predicted_per_second'] for t in timing)]
result={'paired_responses':len(responses),'equal_responses':sum(x['tokens_equal'] and x['text_equal'] for x in responses),'layer_placement_equal':mb['layer_placement']==ma['layer_placement'],'model_buffers_equal':mb['model_buffers']==ma['model_buffers'],'gpu_order_equal':mb['gpu_order']==ma['gpu_order'],'boot_id_equal':mb['boot_id']==ma['boot_id'],'warm_samples_per_version':5,**metrics,'prefill_change_percent':100*(metrics['after']['prompt_per_second']/metrics['before']['prompt_per_second']-1),'decode_change_percent':100*(metrics['after']['predicted_per_second']/metrics['before']['predicted_per_second']-1),'responses':responses}
(R/'dsa-validation-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('responses','before','after')},indent=2));print('decode',metrics['before']['predicted_per_second'],metrics['after']['predicted_per_second'])
if result['equal_responses']!=7 or not all(result[k] for k in ('layer_placement_equal','model_buffers_equal','gpu_order_equal','boot_id_equal')):raise SystemExit(1)
