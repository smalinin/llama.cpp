#!/usr/bin/env python3
"""Compare matched requests and draft counters in isolated before/after runs."""
import json
from pathlib import Path
root=Path('/home/sergei/_my_sync/llama_upstream_review/stage2')
rows=[]
for label,name in [('dflash-separated','glm5next'),('dflash-unified','glm5next'),
                   ('glm-mtp-separated','glm5next'),('qwen-mtp-separated','qwen4exp')]:
    before=root/'parallel'/(label+'-before')/(name+'-spec-1')
    after=root/'parallel'/label/(name+'-spec-1')
    for path in sorted(after.glob('*-response.json')):
        old_path=before/path.name
        if not old_path.exists(): continue
        old=json.loads(old_path.read_text());new=json.loads(path.read_text())
        ot,nt=old['tokens'],new['tokens']
        diff=next((i for i,(a,b) in enumerate(zip(ot,nt)) if a!=b),None)
        if diff is None and len(ot)!=len(nt): diff=min(len(ot),len(nt))
        rows.append({'profile':label,'request':path.name,'tokens_equal':ot==nt,
                     'text_equal':old['content']==new['content'],'first_difference_zero_based':diff,
                     'before_timings':old['timings'],'after_timings':new['timings']})
(root/'parallel-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
for label in sorted({row['profile'] for row in rows}):
    selected=[row for row in rows if row['profile']==label]
    print(label,'identical',sum(row['tokens_equal'] and row['text_equal'] for row in selected),'/',len(selected))
    for row in selected:
        a,b=row['before_timings'],row['after_timings']
        print(' ',row['request'],'first diff',row['first_difference_zero_based'],
              'draft/accepted',a.get('draft_n'),a.get('draft_n_accepted'),'->',b.get('draft_n'),b.get('draft_n_accepted'))
