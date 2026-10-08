from pathlib import Path
import hashlib,json,re,sqlite3
R=Path(__file__).resolve().parent
inputs=json.loads((R/'attention-input-summary.json').read_text())['cases']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary={}
for arch in ['ada0','ada2','ampere']:
 conn=sqlite3.connect(R/f'profile-{arch}.sqlite')
 rows=conn.execute('select k.start,k.end,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start').fetchall()
 main=[r for r in rows if 'void flash_attn_ext_f16<' in r[-1]]
 compaction=[r for r in rows if 'flash_attn_mask_to_sparse_indices<' in r[-1]]
 cases=[json.loads(l) for l in (R/f'attention-profile-{arch}/results.jsonl').read_text().splitlines()]
 assert len(main)==2*len(cases)
 result=[];previous=-1;compact_count=0
 for i,case in enumerate(cases):
  label=case['case'];calls=main[2*i:2*i+2]
  a=R/f'attention-{arch}'/f'{label}.f32';b=R/f'attention-profile-{arch}'/a.name
  assert sha(a)==sha(b)
  kernel=calls[0][-1].replace('(int)','').replace('(bool)0','false').replace('(bool)1','true')
  flag=re.search(r'flash_attn_ext_f16<512, 512, 1, 8, (true|false), (true|false), (true|false)>',kernel)
  assert flag,calls[0][-1]
  sparse=flag.group(3)=='true';assert sparse==inputs[label]['sparse_from_predicate']
  assert calls[0][2:]==calls[1][2:]
  compact=[r for r in compaction if previous<r[0]<calls[-1][0]]
  assert len(compact)==(2 if sparse else 0),(label,len(compact),sparse)
  previous=calls[-1][0];compact_count+=len(compact)
  result.append({**case,'profile_output_bit_exact':True,'output_sha256':sha(a),'sparse_kernel':sparse,'main_grid':calls[0][2:5],'main_block':calls[0][5:8],'main_kernel':calls[0][-1],'compaction_calls':len(compact)})
 assert compact_count==len(compaction)
 summary[arch]={'cases':result,'main_calls':len(main),'compaction_calls':compact_count}
(R/'profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({a:{'cases':len(s['cases']),'main_calls':s['main_calls'],'compaction_calls':s['compaction_calls'],'sparse_cases':sum(c['sparse_kernel'] for c in s['cases'])} for a,s in summary.items()},indent=2))
