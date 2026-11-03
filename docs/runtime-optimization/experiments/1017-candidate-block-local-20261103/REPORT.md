# DS41 block keep transport/local expansion, 2026-11-03

**Согласованный прототип реализован и проверен; по отдельному запросу пользователя теперь default ON,0 отключает.** Base `my_maya b3377774b`. [Одобренный дизайн](../../designs/1017-candidate-block-local.md). Исходные измерения выполнены до promotion; отдельные default-checks описаны в конце. Deployment не выполнялся. Модели, пользовательские presets и my_build не изменялись.

## Изменение

`LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1` фиксируется при создании контекста. Graph params/reuse учитывают eligibility. Graph builder до backend assignments проверяет source index-key buffer, реальные query projection devices, source mappings и support всех выбранных потребителей. Текущий F32 keep (0/-inf на блок) передаётся компактно, прежние REPEAT/reshape/trim создают полную маску локально. Build-local map разделяет tensor identity в пределах одного candidate domain/device: query24/28 на CUDA3 и32/36 на CUDA4. Это наблюдаемое размещение fixture, не hardcoded таблица GPU.

Legacy mask остаётся fallback, но не включается в executable eligible graph. Новых kernels/streams/events/persistent buffers/ручных data pointers/изменений scheduler нет. Сохранены source score, -inf padding, newest-block +inf pin, max/TopK/SET_ROWS, strides и независимый final TopK2048. Candidate2048blocks x8 не заменяется source finalTopK.

Scope: DS41/source20/ratio1/block8/candidate2048/final2048, causal fused LID/offloaded KV/CUDA layer split, P1/text/one stream,6<Q<=512,без eval/abort callback и LoRA. Неподходящие формы/owners/support сохраняют исходный путь. Установка флага после загрузки не является интерфейсом runtime. Предыдущий CANDIDATE_POOL_LOCAL остаётся независимым и defaultOFF; в основных A/B он фиксирован1 на всех arms.

## Точность и lifecycle

- External operator: 98 случаев на двух фактически проверенных GPU; целые mask/block/TopK bytes exact. Пограничные N16384/16385/16391/16392 и long98305/172031, Q1/6/7/511/512/513, finite/all-inf/ties/NaN+inf, padding/невидимые строки/newest block. Этот harness проверяет операции, не production eligibility всех режимов.
- Fixed32 changing inputs на свежих contexts: обе GPU, оба режима, eager1/capture1/launch31/update_failures0. Production full-prefix прогон отдельно упражняет многократную ротацию pipeline slots.
- Отдельный external operator failure/retry: test-only backend wrapper прерывает submission CUDA1 после предыдущего split; обычный scheduler sync завершает ожидающую работу, retry даёт exact mask/TopK/block outputs в обоих режимах. Это не новый full-model KV abort contract.
- Два отдельных свежих процесса OFF/ON, по два полных98K корпуса text/code: 24 raw groups exact между current OFF/ON и сохранённым эталоном. Target/draft logits, taps37/38/39, seq/global/partial states, K3/K5/confidence, append1/7/511/513, restore/resume, rollback zero/partial/full acceptance exact. NLL128 labels/corpus/arm delta0; это continuation diagnostic, не общая оценка качества модели.
- 4 controlled seed failures: очистка обоих contexts и full-prefix recovery exact. Оба exit0 без allocation/compute-size warnings, OOM/CUDA errors/forcedkill; minimums 1325.25, 1325.25 MiB.
- Read-only activation: OFF local0/legacy166; ON local166graphs/332repeats, legacywide0. Новая ветвь реально исполнялась.
- Production no_alloc negative controls P2 и finalTopK1024: OFF/ON graph placements и budgets exact, local nodes0. Pool0/block0-1 metadata controls подтверждают отдельный dispatch; это не новый native speed/quality gate для pool0.

[Raw gates и manifests](artifacts/precision-results.json), [operator](artifacts/operator-summary.json), [negative controls](artifacts/negative-results.json).

## Расчёт памяти и фактический контекст

Model/KV budgets и все15360 routed bundles GPU/actual routes прежние. Меняется compute reserve:

| Context | Combined delta, MiB | Изменившиеся GPU, MiB |
| ---: | ---: | --- |
| 100352 | -294.00 | CUDA3 +196.00, CUDA4 -490.00 |
| 176128 | -520.52 | CUDA3 +344.00, CUDA4 -864.52 |

Начальные metadata пересекались с external operator по времени. Их compute estimates сохранены, но free-before для допуска не использованы. Перед native A/B metadata сняты заново на свободных GPU; исходные файлы сохранены в metadata-initial.

Actual ctx176128:8192+256,172032+16,повтор8192+16,cache_n0. Placement observer подтвердил15360GPU bundles/host routed0 и per-tensor match к no_alloc. IDs exact к control/historical long и повторному prefix. Sampled minimum **1143.25 MiB** на каждой GPU >=512; exit0 без compute mismatch/OOM/CUDA errors/fallback/forcedkill. Уменьшать контекст не потребовалось. Это не auto-fit262K proof. NVML100ms/guard200ms не гарантируют отсутствие пика между samples.

[Metadata](artifacts/metadata-results.json), [admission](artifacts/payload-admission.json), [actual run](artifacts/payload-176128/results.json).

## Clean native OFF/ON/OFF

Q3_K-AntirezLike10shards/MXFP8Engram/Q8KV/BF16DSparkK3/full/p_min0.5/backend samplingON,ctx100352/B8192/U512/P1/TopK2048,прежний frozen allGPU split/overrides,fit/cache/staging/mmprojOFF. HC/M2/LOCAL defaultsON,ASYNCoff,previouspool1 во всех arms. Полные raw IDs и expanded argv/env в manifest. Нет profiler/LD_PRELOAD/eval callback/calibration/builds в timed фазе.

Три отдельных процесса,27 uncached запросов по256greedy:512warm+2,8Kwarm+3,98Kwarm+1 measured на arm. OFF в таблице - среднее двух OFF-медиан. Весь payload, source/binary и tensor placement фиксированы.

| Prefix | OFF PP,t/s | ON PP,t/s | DeltaPP | OFF TG,t/s | ON TG,t/s | DeltaTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 564.55 | 568.89 | +0.77% | 43.46 | 43.54 | +0.19% |
| 8192 | 1081.03 | 1085.81 | +0.44% | 48.45 | 48.53 | +0.16% |
| 98304 | 1121.83 | 1154.72 | +2.93% | 46.23 | 45.94 | -0.62% |

Все27 arrays и draft generated/accepted counters exact; model buffers exact. Exit0 у всех, без предупреждений и floor breach. Minimums 1435.25, 1435.25, 1437.25MiB. Один arm order и singleton98K ограничивают выводы о небольших deltas, особенно TG; ranges и OFF drift сохранены. Проценты предыдущего pooling и этого опыта нельзя складывать. Это не matched Maya comparison.

[Полные результаты](artifacts/bench-results.json).

## Copy proof отдельно от speed

На172K request observer сопоставляет полные src/dst pointers, exact bytes/ne/nb/type, ожидаемые CUDA devices, protected destination clone, cur_copy slots, actual REPEAT consumer и stable metadata. Не читает GPU payload, не добавляет buffers/sync и включён только для long request.

- Wide full score20 CUDA2->3: 0.
- Pooled block scores CUDA2->3: 134, 3.337537 GB.
- Wide dense candidate mask CUDA3->4: **0**.
- Compact keep CUDA3->4: **134**, **3.337537 GB**, protected slots[0, 1, 2, 3].

В коротком fallback score/mask остались: 5 requests каждого, 13.090 MB каждого. В aligned wide fixture новое keep payload равно прежним pooled-score bytes и в8раз меньше прежнего dense-mask payload26.700292096GB. Это logical CUDA API bytes, не hardware DMA counters и не8xwhole-engine speedup. Все selected frames имеют stable metadata/status0; unmatched copies исключены и сохранены.

[Анализ](artifacts/audit-analysis-summary.json), [полный proof](artifacts/audit-analysis.json.gz) и [raw trace](artifacts/payload-176128/audit.jsonl.gz).

## Использование и оставшиеся границы

По запросу пользователя флаг **ON при unset/1**,0отключает. Значение фиксируется до загрузки нового контекста. Для воспроизведения измеренного сочетания:

```bash
LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1 ./llama-server ...
```

Явный0возвращает прежний block path;unset разрешает проверенный guarded путь. Это не команда изменения пользовательского preset. Broader models/KV/media/parallel>1, pool0 native и auto-fit262K не следуют из этой серии. Promotion выполнен по отдельному решению пользователя; guards прежние. Дальше по плану matched Maya и Q2/Q4 MMQ; CPU load сама по себе не является доказательством ускорения CPU expert math.

[Артефакты и воспроизведение](artifacts/README.md), [provenance](artifacts/provenance.json), [SHA256](artifacts/SHA256.json). Собственные GPU-процессы завершены.


## Default ON по запросу пользователя

После отдельного запроса пользователя `LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL` разрешён при unset или1;0 отключает. Флаг по-прежнему фиксируется при создании контекста, eligibility/reuse и guards/fallback прежние. `LLAMA_DSV41_CANDIDATE_POOL_LOCAL` не изменён и остаётся opt-in/defaultOFF. Promotion меняет только условие разбора unset, не kernels/операции/размещение/events. llama-server пересобран. В момент promotion commit/deploy не выполнялись.

Fresh production no_alloc:7процессов,ctx100352 сpool0 default/1/0 иpool1 default/1,ctx176128 сpool0 default/1. Default exact кexplicit1 по candidate graph placements/shapes,model/KV/compute budgets и per-tensor routes;0убирает local nodes. Основной новый default сpool0 проверен здесь по metadata, не новым native A/B. Старые24rawgroups/native/copy/memory gates относятся кexplicit1 сpool1 и не выдаются за повторённые послеpromotion. Runtime headers/helper ABI не менялись; source/library identities и однострочный delta сохранены отдельно.

[Проверки promotion](promotion-default-on/results.json), [точный delta](promotion-default-on/promotion.patch), [checksums](promotion-default-on/SHA256.json). Исходные124артефакта предыдущего опыта не переписаны.


## Позднейшее закрытие pool0 native gap

[Отдельный gate](../1017-pool0-default-20261103/REPORT.md) на `0e6f3cec3` проверил actual defaults: 24 raw groups exact, delta NLL=0, четыре seed failures/retry, 27 native requests и actual ctx=163840 с минимумом 679.25 MiB. Исходные pool=1 gates, скорость и actual 176128 этого отчёта остаются историческими результатами. Runtime/defaults не менялись.
