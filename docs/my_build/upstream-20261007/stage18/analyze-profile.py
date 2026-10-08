import hashlib,json,sqlite3
from pathlib import Path
R=Path(__file__).resolve().parent
s={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arch in ['ada','ampere']:
 c=sqlite3.connect(R/f'profile-{arch}.sqlite')
 rows=c.execute('select k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start').fetchall()
 main=[r for r in rows if 'void flash_attn_ext_f16<' in r[-1]]
 fixup=[r for r in rows if 'void flash_attn_stream_k_fixup_uniform<' in r[-1]]
 cases=list(map(json.loads,(R/f'attention-profile-{arch}/results.jsonl').read_text().splitlines()))
 assert len(main)==len(fixup)==2*len(cases)
 records=[]
 for i,case in enumerate(cases):
  label=case['case'];a=R/f'attention-{arch}'/f'{label}.f32';b=R/f'attention-profile-{arch}'/a.name
  assert sha(a)==sha(b)
  calls=main[2*i:2*i+2];assert calls[0]==calls[1]
  assert calls[0][0]==case['nk']//4
  records.append({**case,'profile_output_bit_exact':True,'output_sha256':sha(a),'main_grid':calls[0][:3],'main_block':calls[0][3:6],'main_kernel':calls[0][-1],'fixup_grid':fixup[2*i][:3],'fixup_kernel':fixup[2*i][-1]})
 s[arch]={'cases':records,'main_calls':len(main),'fixup_calls':len(fixup),'conclusion':'Same FA specialization, grid.x64 at nk256 and128 at nk512; padding changes Stream-K partition/fixup arithmetic.'}
(R/'profile-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({a:{'cases':len(v['cases']),'main_calls':v['main_calls'],'fixup_calls':v['fixup_calls'],'grids':sorted({tuple(c['main_grid']) for c in v['cases']})} for a,v in s.items()},indent=2))
