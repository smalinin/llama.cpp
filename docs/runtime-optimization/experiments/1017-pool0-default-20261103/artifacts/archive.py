from pathlib import Path
import json,hashlib,gzip,shutil
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');s=json.load(open(P/'summary.json'));assert s['complete'];cfg=json.load(open(P/'config.json'));E=R/'docs/runtime-optimization/experiments/1017-pool0-default-20261103';A=E/'artifacts';A.mkdir(parents=True,exist_ok=False)
(E/'.gitignore').write_text('# Keep the raw validation logs referenced by manifests.\n!*.log\n')
(E/'.gitattributes').write_text('# Preserve raw whitespace in validation evidence.\nartifacts/**/*.log -whitespace\nartifacts/**/*.patch -whitespace\n')
def copy(f,d):
 if f.stat().st_size>1024*1024 and f.suffix in ['.jsonl','.log']:
  with f.open('rb')as inp,gzip.open(d/(f.name+'.gz'),'wb')as out:shutil.copyfileobj(inp,out)
 else:shutil.copyfile(f,d/f.name)
for f in sorted(P.iterdir()):
 if f.is_file()and f.suffix in ['.py','.cpp','.json','.jsonl','.log','.md']:copy(f,A)
 elif f.is_dir() and (f.name.startswith(('precision-', 'bench-', 'payload-')) or f.name == 'summary-recovery'):
  d=A/f.name;d.mkdir()
  for x in sorted(f.rglob('*')):
   if x.is_file() and x.suffix in ['.json','.jsonl','.log']:
    target=d/x.relative_to(f).parent;target.mkdir(parents=True,exist_ok=True);copy(x,target)
prov={'head':s['head'],'production_sources_changed':False,'source_files':cfg['source_files'],'libraries':cfg['source_library_hashes'],'helpers':{n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['fresh','metadata','trace.so','metadata-trace.so','placement.so']},'helper_binary_locator':str(P),'historical_raw_reference':'/tmp/dsv41-candidate-pool-20261103/fresh-on-native/results.json'};(A/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
rows=[]
for r in s['native']:
 a=r['arms'];offpp=sum(a[k]['PP']for k in ['off-pre','off-post'])/2;offtg=sum(a[k]['TG']for k in ['off-pre','off-post'])/2
 rows.append(f"| {r['n']} | {offpp:.2f} | {a['on']['PP']:.2f} | {r['PP_delta_pct']:+.2f}% | {offtg:.2f} | {a['on']['TG']:.2f} | {r['TG_delta_pct']:+.2f}% |")
mem=s['memory'];admission='\n'.join(f"| {r['context']} | {'да'if r['admitted']else'нет'} | {r['guarded_min_mib']:.2f} |"for r in mem['candidate_admission'])
near='Для near-prefix есть исторический172Kконтроль; IDs совпали.'if mem['same_near_to_historical']is True else'Для выбранного shorter near-prefix нет исторического token oracle; его успешное завершение является lifecycle/memory gate, не новым long-prefix quality доказательством.'
report=f'''# DS41: native проверка текущих candidate defaults, 2026-11-03

**Этап завершён.** Runtime `my_maya {s['head'][:9]}` и binaries заморожены. Runtime/defaults/model files/user presets/my_build не изменялись, нового commit/deploy нет. Это закрытие оставшегося native gate для **POOL_LOCAL unset/OFF и BLOCK_LOCAL unset/ON**. Предыдущие explicit1/pool1 speed и actual176128 результаты остаются отдельной серией.

## Протокол

Q3_K-AntirezLike10shards/MXFP8Engram/Q8KV/BF16DSparkK3/full/p_min0.5/backend samplingON,B8192/U512/P1/TopK2048,20threads,ctx100352 для speed/precision. Frozen all15360GPUexpert bundles и прежний split/overrides,fit/cache/staging/mmprojOFF,defaultHC/M2/LOCAL/ASYNCoff. Под defaults здесь понимаются два candidate-флага; это не заявление о стандартных значениях всех CLI/env runtime. Полные argv/env/rawIDs/hashes в manifests.

Pooling-переменная отсутствует во всех дочерних процессах. У default-ветви BLOCK_LOCAL также отсутствует; контроль задаёт только BLOCK_LOCAL=0. Прежний precision helper принудительно устанавливал pooling1 внутри corpus loop. В новой копии это удалено; source/hash и фактический remote pool owner проверены. Старые результаты не переписаны. Test-only fault injection теперь работает независимо от pool-mode, только по явному запросу DSV41_TAIL_FAULT_AFTER.

## Точность и rollback

Два свежих процесса OFF/default, по два полных98Kкорпуса text/code:24raw result groups exact между режимами и сохранённым эталоном. Target/draft logits, taps37/38/39, seq/global/partialstate, K3/K5/confidence, append1/7/511/513, prefix restore/reuse и zero/partial/full rollback совпали побитово. NLL128 labels/corpus/mode delta0. Это continuation diagnostic, не full-corpus PPL.

Четыре controlled seed failures: и trace injection count, и marker count по2/процесс; очистка обоих contexts и full-prefix recovery exact. Оба exit0безallocation/compute mismatch/OOM/CUDAerrors/forcedkill. Minimums {', '.join(f'{v:.2f}'for v in s['precision']['minimum_free_mib'])}MiB >=512. Traceподтверждает pool remoteCUDA3/sourceCUDA2 в обоих режимах. OFFlocal0/legacy166;default166graphs/332localREPEAT/legacywide0.

[Raw precision](artifacts/precision-results.json), [config/source identity](artifacts/config.json).

## Clean native OFF/default/OFF

Три отдельных процесса,27full uncached requests/cache_n0,256greedy outputs/request.512warm+2,8Kwarm+3,98Kwarm+1measured/arm. Нет profiler/LD_PRELOAD/eval callback/calibration/builds в timed runs. OFF в таблице - среднее двух OFF-медиан.

| Prefix | OFF PP,t/s | Default PP,t/s | DeltaPP | OFF TG,t/s | Default TG,t/s | DeltaTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Все27output arrays и draftgenerated/accepted counters exact; они также совпали с ранее сохранёнными pool1 controls. Это не сравнительный speed gate pool0/pool1: текущая скорость измеряет block OFF/default при pooling OFF. Actual model buffer logs совпали,всеexit0безwarnings/floor breach. Minimums {', '.join(f'{v:.2f}'for v in s['native_minimum_free_mib'])}MiB. Ranges и OFF drift сохранены.98Ksingleton и один порядок arms ограничивают выводы о малых deltas,TG и inactive512/8Kcandidate path. Проценты предыдущих серий не складываются.

[Полный native manifest/results](artifacts/bench-results.json).

## Metadata и выбор безопасного контекста

Fresh idle no_alloc100352OFF/default,176128/163840/159744default. Model/KV budgets и per-tensor routes exact; actual poolCUDA3/source scoreCUDA2,local repeatsCUDA3/4. Для допуска взят max observed per-device overhead текущего100Kbenchmark и historical176Kpool1/block1 с matching metadata, плюс64MiB guard. CUDA/NVML mapping проверен поUUID/PCI. Это conservative estimate, не memory proof.

| Context | Допущен | Guarded minimum free,MiB |
| ---: | --- | ---: |
{admission}

В memory запуске выбран **ctx{mem['context']}**, near-prefix{mem['prefix']}. Pooling не включался ради увеличения контекста,placement не менялся. Это отдельный результат от прежнего pool1actual176128; меньший общий reserve не гарантирует больше свободной памяти на каждой конкретной GPU.

[Metadata commands/env](artifacts/metadata-runs.json), [admission с per-GPU расчётом](artifacts/payload-admission.json).

## Actual memory и lifecycle

Fresh default server:8192+256,{mem['prefix']}+16,повтор8192+16,cache_n0. Loader-only placement observer, без forward/CUDAAPI probes, дополнительных GPUbuffers/sync. Все15360routed bundlesGPU/host routed0; per-tensor placement exact кmetadata. Минимум свободнойVRAM **{mem['minimum_free_mib']:.2f}MiB** на всехGPU >=512;exit0безallocation mismatch/OOM/CUDAerrors/fallback/forcedkill.

8KIDs совпали с control benchmark,повторный8K - с first16. {near} Это не новая speed A/B серия и не новый copyAPI proof. NVML100ms/watcher200ms проверяют sampled floor,не гарантируют отсутствие пика между samples.

[Actual results](artifacts/payload-{mem['context']}/results.json).

## Вывод и следующие задачи

Native gap текущих candidate defaults закрыт в этой frozen Q3/Q8KV/BF16K3/P1конфигурации. Исправление runtime для прохождения gates не потребовалось; defaultBLOCKON иPOOL_OFF прежние. Source/library hashes неизменны. Другие LLM/KV/media/parallel>1/auto-fit262K этим не объявляются проверенными. MatchedMaya, свежий residual profile и Q2/Q4MMQ остаются отдельными этапами; текущие числа не подменяют их.

[Артефакты и воспроизведение](artifacts/README.md), [provenance](artifacts/provenance.json), [SHA256](artifacts/SHA256.json). Все собственные GPU-процессы завершены.\n'''
(E/'REPORT.md').write_text((P/'REPORT-prepared.md').read_text())
(A/'README.md').write_text('''# Reproduction and scope

Full argv/env/rawIDs/source/library identities are in config.json, precision-results.json, metadata-validation.json (corrected semantic validation; raw metadata-runs.json retained incomplete), bench-results.json, payload-admission.json and payload-*/results.json. No production source/default/model mutation. Recompile helpers against matching current internal headers; compiled binaries remain at /tmp/dsv41-pool0-default-20261103, hashes in provenance. Metadata-only checks do not replace actual memory.

run-gates.py is guarded by __main__ and has a distinct name from Python stdlib queue. Never name an executable harness queue.py. Serial phases:prepare.py ->run-precision.py OFF/default ->run-metadata.py ->bench.py OFF/default/OFF ->admit.py ->run-payload.py ->finalize.py. Existing test helper's forcedPOOL1 removed in this copy; trace proves pooling remains remote and unset. Current default BLOCK unset is intentionally not explicit1. Earlier pool1/explicit1 reports remain historical controls, not the speed baseline. No concurrent GPU work during these phases. No automatic commit/default change/deploy.

Large logs/JSONL losslessly gzipped; raw manifests retain /tmp locators. A lower selected context has no previous near-prefix token oracle; it is a memory/lifecycle gate. Report describes exactness scope. Local .gitignore preserves manifested raw .log files despite global ignore; .gitattributes preserves raw whitespace. Post-test summary import replayed metadata because the former queue.py shadowed stdlib queue. summary-recovery preserves diagnostics and snapshots; metadata-validation.json compares all per-device admission inputs and routes. Native, precision and payload result manifests are byte-identical to recovery snapshots (raw-results-integrity.json). Original metadata elapsed timings/hashes cannot be restored; original admission references and incomplete runner manifest are retained. Neither rewrites evidence. SHA256 manifest covers every raw artifact in this directory except itself.
''')
(A/'SHA256.json').write_text(json.dumps({str(f.relative_to(A)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(A.rglob('*'))if f.is_file()},indent=2)+'\n');print('ARCHIVED',E,flush=True)
