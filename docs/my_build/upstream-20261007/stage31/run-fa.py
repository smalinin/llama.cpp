from pathlib import Path
import json,os,subprocess,struct
R=Path(__file__).resolve().parent;out=R/'fa-results';out.mkdir(exist_ok=True);results=[]
cases=[x['label'] for x in json.loads((R/'fa-input-manifest.json').read_text())]
for gpu in [0,3]:
 for q8 in [0,1]:
  ref=None
  for label in cases:
   for ordered in [0,1]:
    name=f'g{gpu}-q{q8}-{label}-o{ordered}';cmd=[str(R/'fa-layout'),str(R/'fa-inputs'/label),str(out/name),str(gpu),str(ordered),str(q8),'100'];env=os.environ.copy();env['LD_LIBRARY_PATH']=str(R.parent/'stage30/preclear-bin')
    with (out/(name+'.log')).open('w') as log:p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=log,text=True)
    if p.returncode:raise RuntimeError(name+': '+(out/(name+'.log')).read_text()[-4000:])
    row=json.loads(p.stdout.strip().splitlines()[-1]);data=(out/(name+'.f32')).read_bytes()
    if ordered and label=='capture0':ref=data
    if ordered:row['equal_ordered_reference']=data==ref;assert data==ref,name
    row.update(label=label,gpu=gpu,command=cmd);results.append(row);print(name,row.get('equal_ordered_reference'),row['us'],flush=True)
    (R/'fa-results.json').write_text(json.dumps(results,indent=2)+'\n')
print('All ordered layouts exact',flush=True)
