from pathlib import Path
import json,hashlib,gzip,shutil,subprocess
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');s=json.load(open(P/'summary.json'));assert s['complete'];E=R/'docs/runtime-optimization/experiments/1017-candidate-block-local-20261103';A=E/'artifacts';A.mkdir(parents=True,exist_ok=False)
excluded={'audit-analysis.json'}
for f in sorted(P.iterdir()):
 if f.is_file()and f.suffix in ['.cpp','.py','.json','.jsonl','.log','.md','.patch']and f.name not in excluded:
  if f.stat().st_size>1024*1024 and f.suffix in ['.jsonl','.log']:
   with f.open('rb')as inp,gzip.open(A/(f.name+'.gz'),'wb')as out:shutil.copyfileobj(inp,out)
  else:shutil.copyfile(f,A/f.name)
for dirname in ['precision-0','precision-1','bench-off-pre','bench-on','bench-off-post','payload-176128','metadata-initial']:
 d=P/dirname;t=A/dirname;t.mkdir()
 for f in sorted(d.iterdir()):
  if not f.is_file()or f.suffix not in ['.json','.jsonl','.log']:continue
  if f.stat().st_size>1024*1024 and f.suffix in ['.jsonl','.log']:
   with f.open('rb')as inp,gzip.open(t/(f.name+'.gz'),'wb')as out:shutil.copyfileobj(inp,out)
  else:shutil.copyfile(f,t/f.name)
copy=json.load(open(P/'audit-analysis.json'));(A/'audit-analysis-summary.json').write_text(json.dumps({k:v for k,v in copy.items()if k!='matches'},indent=2)+'\n')
with(P/'audit-analysis.json').open('rb')as inp,gzip.open(A/'audit-analysis.json.gz','wb')as out:shutil.copyfileobj(inp,out)
prov={'base':s['base'],'source_files':{n:hashlib.sha256((R/n).read_bytes()).hexdigest()for n in ['src/llama-context.cpp','src/llama-context.h','src/llama-graph.h','src/models/models.h','src/models/deepseek41.cpp']},'libraries':s['libraries'],'helpers':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in P.iterdir()if f.is_file()and f.suffix in ['.so']or f.is_file()and f.name in ['fresh','metadata','operator']},'patch':'FINAL.patch','built_helpers_locator':str(P),'models_and_user_presets_unchanged':True};(A/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
rows=[]
for r in s['native']:
 arms=r['arms'];pp=(arms['off-pre']['PP']+arms['off-post']['PP'])/2;tg=(arms['off-pre']['TG']+arms['off-post']['TG'])/2
 rows.append(f"| {r['n']} | {pp:.2f} | {arms['on']['PP']:.2f} | {r['PP_delta_pct']:+.2f}% | {tg:.2f} | {arms['on']['TG']:.2f} | {r['TG_delta_pct']:+.2f}% |")
mt='\n'.join(f"| {r['context']} | {r['sum_compute_delta_mib']:.2f} | "+', '.join(f"{d['device']} {d['compute_delta_mib']:+.2f}"for d in r['delta']if d['compute_delta_mib'])+' |' for r in s['metadata'])
a=s['copy_proof']['wide'];sm=s['copy_proof']['small_fallback'];small_note=f"В коротком fallback score/mask остались: {sm['score20']['calls']} requests каждого, {sm['score20']['logical_api_bytes']/1e6:.3f} MB каждого."
report=f'''# DS41 block keep transport/local expansion, 2026-11-03

**Согласованный прототип реализован и проверен; новый флаг остаётся OFF по умолчанию.** Base `my_maya {s['base'][:9]}`. [Одобренный дизайн](../../designs/1017-candidate-block-local.md). Нового commit/default promotion/deploy нет. Модели, пользовательские presets и my_build не изменялись.

## Изменение

`LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1` фиксируется при создании контекста. Graph params/reuse учитывают eligibility. Graph builder до backend assignments проверяет source index-key buffer, реальные query projection devices, source mappings и support всех выбранных потребителей. Текущий F32 keep (0/-inf на блок) передаётся компактно, прежние REPEAT/reshape/trim создают полную маску локально. Build-local map разделяет tensor identity в пределах одного candidate domain/device: query24/28 на CUDA3 и32/36 на CUDA4. Это наблюдаемое размещение fixture, не hardcoded таблица GPU.

Legacy mask остаётся fallback, но не включается в executable eligible graph. Новых kernels/streams/events/persistent buffers/ручных data pointers/изменений scheduler нет. Сохранены source score, -inf padding, newest-block +inf pin, max/TopK/SET_ROWS, strides и независимый final TopK2048. Candidate2048blocks x8 не заменяется source finalTopK.

Scope: DS41/source20/ratio1/block8/candidate2048/final2048, causal fused LID/offloaded KV/CUDA layer split, P1/text/one stream,6<Q<=512,без eval/abort callback и LoRA. Неподходящие формы/owners/support сохраняют исходный путь. Установка флага после загрузки не является интерфейсом runtime. Предыдущий CANDIDATE_POOL_LOCAL остаётся независимым и defaultOFF; в основных A/B он фиксирован1 на всех arms.

## Точность и lifecycle

- External operator: {s['operator_cases']} случаев на двух фактически проверенных GPU; целые mask/block/TopK bytes exact. Пограничные N16384/16385/16391/16392 и long98305/172031, Q1/6/7/511/512/513, finite/all-inf/ties/NaN+inf, padding/невидимые строки/newest block. Этот harness проверяет операции, не production eligibility всех режимов.
- Fixed32 changing inputs на свежих contexts: обе GPU, оба режима, eager1/capture1/launch31/update_failures0. Production full-prefix прогон отдельно упражняет многократную ротацию pipeline slots.
- Отдельный external operator failure/retry: test-only backend wrapper прерывает submission CUDA1 после предыдущего split; обычный scheduler sync завершает ожидающую работу, retry даёт exact mask/TopK/block outputs в обоих режимах. Это не новый full-model KV abort contract.
- Два отдельных свежих процесса OFF/ON, по два полных98K корпуса text/code: {s['precision_groups']} raw groups exact между current OFF/ON и сохранённым эталоном. Target/draft logits, taps37/38/39, seq/global/partial states, K3/K5/confidence, append1/7/511/513, restore/resume, rollback zero/partial/full acceptance exact. NLL128 labels/corpus/arm delta0; это continuation diagnostic, не общая оценка качества модели.
- {s['faults']} controlled seed failures: очистка обоих contexts и full-prefix recovery exact. Оба exit0 без allocation/compute-size warnings, OOM/CUDA errors/forcedkill; minimums {', '.join(f'{x:.2f}'for x in s['precision_min_mib'])} MiB.
- Read-only activation: OFF local0/legacy166; ON local166graphs/332repeats, legacywide0. Новая ветвь реально исполнялась.
- Production no_alloc negative controls P2 и finalTopK1024: OFF/ON graph placements и budgets exact, local nodes0. Pool0/block0-1 metadata controls подтверждают отдельный dispatch; это не новый native speed/quality gate для pool0.

[Raw gates и manifests](artifacts/precision-results.json), [operator](artifacts/operator-summary.json), [negative controls](artifacts/negative-results.json).

## Расчёт памяти и фактический контекст

Model/KV budgets и все15360 routed bundles GPU/actual routes прежние. Меняется compute reserve:

| Context | Combined delta, MiB | Изменившиеся GPU, MiB |
| ---: | ---: | --- |
{mt}

Начальные metadata пересекались с external operator по времени. Их compute estimates сохранены, но free-before для допуска не использованы. Перед native A/B metadata сняты заново на свободных GPU; исходные файлы сохранены в metadata-initial.

Actual ctx{s['payload']['context']}:8192+256,172032+16,повтор8192+16,cache_n0. Placement observer подтвердил15360GPU bundles/host routed0 и per-tensor match к no_alloc. IDs exact к control/historical long и повторному prefix. Sampled minimum **{s['payload']['min_mib']:.2f} MiB** на каждой GPU >=512; exit0 без compute mismatch/OOM/CUDA errors/fallback/forcedkill. Уменьшать контекст не потребовалось. Это не auto-fit262K proof. NVML100ms/guard200ms не гарантируют отсутствие пика между samples.

[Metadata](artifacts/metadata-results.json), [admission](artifacts/payload-admission.json), [actual run](artifacts/payload-176128/results.json).

## Clean native OFF/ON/OFF

Q3_K-AntirezLike10shards/MXFP8Engram/Q8KV/BF16DSparkK3/full/p_min0.5/backend samplingON,ctx100352/B8192/U512/P1/TopK2048,прежний frozen allGPU split/overrides,fit/cache/staging/mmprojOFF. HC/M2/LOCAL defaultsON,ASYNCoff,previouspool1 во всех arms. Полные raw IDs и expanded argv/env в manifest. Нет profiler/LD_PRELOAD/eval callback/calibration/builds в timed фазе.

Три отдельных процесса,27 uncached запросов по256greedy:512warm+2,8Kwarm+3,98Kwarm+1 measured на arm. OFF в таблице - среднее двух OFF-медиан. Весь payload, source/binary и tensor placement фиксированы.

| Prefix | OFF PP,t/s | ON PP,t/s | DeltaPP | OFF TG,t/s | ON TG,t/s | DeltaTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Все27 arrays и draft generated/accepted counters exact; model buffers exact. Exit0 у всех, без предупреждений и floor breach. Minimums {', '.join(f'{x:.2f}'for x in s['native_min_mib'])}MiB. Один arm order и singleton98K ограничивают выводы о небольших deltas, особенно TG; ranges и OFF drift сохранены. Проценты предыдущего pooling и этого опыта нельзя складывать. Это не matched Maya comparison.

[Полные результаты](artifacts/bench-results.json).

## Copy proof отдельно от speed

На172K request observer сопоставляет полные src/dst pointers, exact bytes/ne/nb/type, ожидаемые CUDA devices, protected destination clone, cur_copy slots, actual REPEAT consumer и stable metadata. Не читает GPU payload, не добавляет buffers/sync и включён только для long request.

- Wide full score20 CUDA2->3: {a['score20']['calls']}.
- Pooled block scores CUDA2->3: {a['pool_scores']['calls']}, {a['pool_scores']['logical_api_bytes']/1e9:.6f} GB.
- Wide dense candidate mask CUDA3->4: **{a['dense_mask']['calls']}**.
- Compact keep CUDA3->4: **{a['compact_keep']['calls']}**, **{a['compact_keep']['logical_api_bytes']/1e9:.6f} GB**, protected slots{a['compact_keep']['destination_slots']}.

{small_note} В aligned wide fixture новое keep payload равно прежним pooled-score bytes и в8раз меньше прежнего dense-mask payload26.700292096GB. Это logical CUDA API bytes, не hardware DMA counters и не8xwhole-engine speedup. Все selected frames имеют stable metadata/status0; unmatched copies исключены и сохранены.

[Анализ](artifacts/audit-analysis-summary.json), full proof в audit-analysis.json.gz и raw audit.jsonl.gz.

## Использование и оставшиеся границы

Флаг **OFF по умолчанию**, включать до загрузки нового контекста. Для воспроизведения измеренного сочетания:

```bash
LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1 ./llama-server ...
```

Unset/0 возвращают прежний block path. Это не команда изменения пользовательского preset. Broader models/KV/media/parallel>1, pool0 native и auto-fit262K не следуют из этой серии. Promotion требует отдельного решения пользователя. Дальше по плану matched Maya и Q2/Q4 MMQ; CPU load сама по себе не является доказательством ускорения CPU expert math.

[Артефакты и воспроизведение](artifacts/README.md), [provenance](artifacts/provenance.json), checksumsSHA256.json. Собственные GPU-процессы завершены.
'''
(E/'REPORT.md').write_text(report)
(A/'README.md').write_text('''# Reproduction and artifact scope

Base and exact five-source delta: config.json / FINAL.patch / provenance.json. Built helper binaries and uncompressed large probes remain at /tmp/dsv41-candidate-block-local-20261103; helper sources, arguments and results are archived here. Recompile against matching current internal headers and libraries; do not run historical helpers blindly against another branch. source/library hashes identify the measured frozen build.

Serial phases: build -> external operator -> fresh OFF/ON precision -> idle no_alloc/negative controls -> clean native OFF/ON/OFF -> guarded actual176128/read-only copy proof -> analyze-audit.py/finalize.py. memory_monitor.py maps CUDA/NVML by UUID/PCI, sampled floor512MiB. Full argv/env/model paths/raw IDs and corpus inputs are in config/precision/bench manifests. Tests must not overlap another GPU workload. No commit, model mutation, default promotion or deployment is implied.

Large JSONL/log files are losslessly gzipped; JSON results retain their exact contents. metadata-initial retains first free-before samples that overlapped the operator; only fresh idle metadata informed admission. pool0 controls are metadata-only. Copy proof is logical API requests, not physical DMA. See REPORT for limitations. JSON paths inside raw manifests refer to the original /tmp run and have not been rewritten.
''')
hashes={str(f.relative_to(A)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(A.rglob('*'))if f.is_file()};(A/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n');print('ARCHIVED',E,'files',len(hashes))
