#!/usr/bin/env python3
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parent
p=ROOT/'controls-ada-profile'
files=list((ROOT/'controls-ada').glob('*.f32'))+list((ROOT/'controls-ada').glob('*.i32'))
for f in files:assert f.read_bytes()==(p/f.name).read_bytes()
assert (ROOT/'controls-ada/results.jsonl').read_bytes()==(p/'results.jsonl').read_bytes()
c=sqlite3.connect(ROOT/'controls-ada-profile.sqlite')
r=c.execute('''select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k
               join StringIds s on s.id=k.demangledName group by s.value order by count(*) desc''').fetchall()
launches=c.execute('''select s.value,k.blockX,k.blockY,k.blockZ,k.gridX,k.gridY,k.gridZ,count(*)
                    from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
                    where s.value like '%mul_mat_vec_q%' group by 1,2,3,4,5,6,7''').fetchall()
scalar=[x for x in r if 'mul_mat_vec_q<(ggml_type)10, (int)1, (bool)1' in x[0]]
wide=[x for x in r if 'mul_mat_vec_q_moe<(ggml_type)10, (int)2, (bool)1' in x[0]]
assert len(scalar)==len(wide)==1 and scalar[0][1]==60 and wide[0][1]==16
out={'profile_output_files_match':len(files),'profile_results_identical':True,
     'note':'Both modes in one profile. Scalar Q2_K fused up/gate calls include 6 native scalar and 54 scalarized calls; 16 wide Q2_K calls are native controls. No benchmark in this process.',
     'kernels':[dict(name=x[0],calls=x[1],gpu_ms=x[2]) for x in r],
     'launches':[dict(name=x[0],block=list(x[1:4]),grid=list(x[4:7]),calls=x[7]) for x in launches]}
(ROOT/'kernel-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['kernels','launches']}))
