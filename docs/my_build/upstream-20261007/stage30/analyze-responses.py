from pathlib import Path
import json,argparse
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('label');a=p.parse_args();root=R/'server-runs'/a.label;d=next(x.parent for x in root.rglob('result.json'))
rows=[]
for i in range(3):
 baseline=json.loads((d/f'fresh1-{i}-response.json').read_text())
 for phase in ['reuse1','reuse2','fresh2','resident-fresh1','resident-reuse1','resident-reuse2','resident-fresh2']:
  f=d/f'{phase}-{i}-response.json'
  if not f.exists():continue
  cur=json.loads(f.read_text());aa=baseline['tokens'];bb=cur['tokens'];first=next((j for j,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
  if first is None and len(aa)!=len(bb):first=min(len(aa),len(bb))
  rows.append(dict(prompt=i,phase=phase,tokens_equal=aa==bb,first_different_token=first,text_equal=baseline['content']==cur['content'],available_probabilities_equal=baseline['completion_probabilities']==cur['completion_probabilities'],cache_n=cur['timings']['cache_n'],prompt_n=cur['timings']['prompt_n']))
summary={'label':a.label,'comparisons':rows};(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
