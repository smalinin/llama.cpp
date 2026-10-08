from pathlib import Path
import importlib.util,json
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',R/'analyze-chain.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
result={}
for target in [0,86]:
 orders={}
 for width in [1,2,4]:
  d=a.D/f'{a.MODE}-w{width}-input{target}'
  col=json.loads((d/'alignment.json').read_text())['column']
  tensors=[json.loads(l) for l in (d/'tensors.jsonl').read_text().splitlines()]
  orders[width]={int(t['owner'].rsplit('-',1)[1]):list(map(int,a.read_tensor(d,t,1,col))) for t in tensors if t['owner'].startswith('ffn_moe_topk-') and t['role']=='output'}
 result[target]={str(width):{'order_differences':[layer for layer in range(40) if orders[1][layer]!=orders[width][layer]],'set_differences':[layer for layer in range(40) if sorted(orders[1][layer])!=sorted(orders[width][layer])]} for width in [2,4]}
(R/'expert-selection-summary.json').write_text(json.dumps(result,indent=2)+'\n')
