#!/usr/bin/env python3
import json
from pathlib import Path
import sqlite3
ROOT=Path(__file__).resolve().parent
out={'profile_outputs_exact':0,'runs':{}}
for dataset in ['hc0','router0']:
 for arch in ['ada','ampere']:
  label=f'{dataset}-{arch}';p=ROOT/(label+'-profile');files=list((ROOT/label).glob('*.f32'))
  for f in files:assert f.read_bytes()==(p/f.name).read_bytes()
  assert (ROOT/label/'results.jsonl').read_bytes()==(p/'results.jsonl').read_bytes()
  out['profile_outputs_exact']+=len(files)
  c=sqlite3.connect(ROOT/(label+'-profile.sqlite'))
  rows=c.execute('''select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value order by count(*) desc''').fetchall()
  launches=c.execute('''select s.value,k.blockX,k.blockY,k.blockZ,k.gridX,k.gridY,k.gridZ,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by 1,2,3,4,5,6,7''').fetchall()
  out['runs'][label]={'files_exact':len(files),'kernels':[{'name':x[0],'calls':x[1],'gpu_ms':x[2]} for x in rows],'launches':[{'name':x[0],'block':list(x[1:4]),'grid':list(x[4:7]),'calls':x[7]} for x in launches]}
assert out['profile_outputs_exact']==56
(ROOT/'kernel-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
