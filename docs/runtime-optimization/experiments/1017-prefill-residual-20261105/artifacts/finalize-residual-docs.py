from pathlib import Path
import json,hashlib,shutil,gzip,subprocess
R=Path('/home/sergei/Github/llama.cpp');K=R/'docs/runtime-optimization';E=K/'experiments/1017-prefill-residual-20261105';P=E/'artifacts';D=P/'shared-local'
def read(p):return json.loads(p.read_text())
def save(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
j=read(D/'checks.json');assert j['complete']and j['all_outputs_counters_exact_to_baseline']and j['reverse_transition_before']==24 and j['reverse_transition_after']==0 and j['gain_percent']<1
p=E/'REPORT.md';s=p.read_text().replace('A placement-only shared-expert screen is pending below.','The placement-only shared-expert screen also completed without a useful PP gain.').replace('Admission and actual result are recorded separately below.','Admission and actual result are recorded below.')
insert='''## 6. Placement-only shared gate/up co-location, negative screen

The existing override CLI moves only `blk.6.ffn_(gate|up)_shexp.weight` toCUDA1 and `blk.29.ffn_(gate|up)_shexp.weight` toCUDA4. All six shared weights areQ8_0, each12533760bytes. Down was already on the destination card. Routed expert placement/context and all15360 GPU bundles remain exact. Production copies4, launch queue unset, CPU lookup early default ON. Metadata and actual memory admission pass without reducing context.

| 8K native pre/post screen | Baseline | Shared gate/up local |
| --- | ---: | ---: |
| PP pre, t/s | 1436.28 | 1428.49 |
| PP post, t/s | 1425.42 | 1414.47 |
| Mean PP, t/s | 1430.85 | 1421.48 |
| Delta from baseline | - | -0.65% |
| Observed reverse transitions before shared work | 24 | 0 |
| Target launch calls in armed request | 51068 | 49634 |

Activation is observed, not inferred only from CLI: the24 target-ubatch backward no-input transitions CUDA1->CUDA0 andCUDA4->CUDA3 disappear. The16 CUDA3->CUDA4 transitions before KV SET_ROWS remain protected. Despite fewer graph/launch calls, native PP does not improve. This rejects only these two shared-expert boundaries as a sufficient explanation, not every possible placement or dependency improvement.

All five same-prefix outputs and draft counters match baseline, actual minimum1323.25MiB, exit0 and no warnings/floor breach. No expanded raw-logit/state/rollback/NLL run or default/preset change is justified by this negative speed screen. TG16 remains diagnostic, not a sustained TG comparison. [Checks and transition counters](artifacts/shared-local/checks.json), [metadata/admission](artifacts/shared-local/admission.json), [API summary](artifacts/shared-local/summary.json), [host summary](artifacts/shared-local/host-summary.json), [lossless trace](artifacts/shared-local/scopes.jsonl.gz).

Next prefill step: audit actual producer/view-root/buffer ownership and slot lifetime around remaining KV writes and cross-device reuse. The current first/last summaries are insufficient to authorize an event-order rewrite. A new scheduler pattern must have a reviewable ownership design before implementation. Separate native verification sizes1..6 and prior Q2/Q4/media/parallel/auto-fit tasks remain open.

'''
s=s.replace('## Evidence and limits\n',insert+'## Evidence and limits\n');s=s.replace('The first five phases contain29 full-prefix uncached requests including warmups;','All six phases contain34 full-prefix uncached requests including warmups;');p.write_text(s)
for f in [K/'README.md',K/'CATALOG.md',K/'models/deepseek41-1017.md']:
 t=f.read_text();t=t.replace('A small existing-CLI shared-expert co-location screen follows in the same report; no scheduler reorder is implied.','The existing-CLI shared gate/up6/29 co-location screen removed24 backward transitions but PP changed-0.65%; it is not adopted. No scheduler reorder is implied.');t=t.replace('Shared-expert placement screen is recorded separately in that report.','Shared gate/up6/29 co-location removed24 backward transitions but PP changed-0.65%, so it is not adopted.');t=t.replace('Shared-expert co-location via existing overrides is a separate screen in that report.','Shared gate/up6/29 co-location removed24 backward transitions but PP changed-0.65%, so it is not adopted.');f.write_text(t)
p=K/'LESSONS.md';p.write_text(p.read_text()+'\nThe shared-expert placement screen removed all24 observed backward transitions at boundaries6/29 and cut launch count51068->49634, yet native PP changed-0.65%. Confirming activation does not establish a useful engine gain. The retained KV SET_ROWS path is real work; the next audit must establish writer/root-buffer/slot ownership before any scheduling change.\n')
for name in ['DEEPSEEK_V41_1017_MAYA_CACHE_STAGE5_DESIGN.md','DEEPSEEK_V41_1017_MAYA_EXPERT_CACHE_PLAN.md']:
 p=Path('/home/sergei/_my_old/_my_deepseek41')/name;t=p.read_text();old='- [ ] Finish placement-only shared gate/up6->CUDA1 and29->CUDA4 screen, same down projections and routed expert layout. Metadata admission passed; actual monitored request results are pending in the report. No user preset change or adoption from a screen.';assert old in t;t=t.replace(old,'- [x] Placement-only shared gate/up6->CUDA1 and29->CUDA4 screen finished: backward transitions24->0, PP1430.85->1421.48t/s (-0.65%), actual minimum1323.25MiB, five outputs/counters exact, exit0. No gain/adoption or user preset change. All six residual phases have34 same-prefix exact requests.');p.write_text(t)
shutil.copy2('/tmp/residual-source-lifetime.json',P/'source-lifetime-audit.json');shutil.copy2('/tmp/finalize-residual-docs.py',P/'finalize-residual-docs.py');shutil.copy2('/tmp/update-residual-knowledge.py',P/'update-residual-knowledge.py')
state=read(P/'STATE.json');state.update(status='Complete: six residual phases,34 same-prefix exact requests. Queue4x, private copies8 and shared gate/up6/29 co-location yield no useful PP gain; none adopted. Production/default/preset/model unchanged; GPU released.',next=['Read REPORT.md and separate copies8 recovery checks; no candidate promotion or commit.','Next prefill stage: read-only producer/root-buffer/slot ownership audit around retained KV writes and cross-device reuse, then reviewed design if new scheduler ordering is warranted.','Separate six-query verification and previous open tasks remain open.']);save(P/'STATE.json',state)
# Revalidate evidence parsing, compression and versionable files without touching raw manifests.
files=subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(E)],cwd=R,text=True).splitlines();inventory=[]
for rel in files:
 f=R/rel
 if f.name=='evidence-manifest.json':continue
 data=f.read_bytes();assert data[:4]!=b'\x7fELF',f
 if f.suffix=='.json':json.loads(data)
 if f.suffix=='.py':compile(data,str(f),'exec')
 if f.name.endswith('.jsonl.gz'):
  raw=f.with_suffix('');assert gzip.decompress(data)==raw.read_bytes()
 inventory.append({'path':str(f.relative_to(E)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
save(P/'evidence-manifest.json',{'complete':True,'files':inventory,'note':'Versionable experiment evidence only; excludes this manifest, ignored raw JSONL, compiled binaries and private objects. Original incomplete copies8 controller retained and separately recovered. Production library/source unchanged.'})
subprocess.run(['git','diff','--check'],cwd=R,check=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print('CLOSED',len(inventory),'versionable evidence files; no GPU processes; no runtime changes.')
