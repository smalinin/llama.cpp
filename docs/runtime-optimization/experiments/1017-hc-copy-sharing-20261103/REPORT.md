# HC root/reshape sharing: prototype and validation, 2026-11-03

**Прототип реализован и проверен: PP +3.8..4.3%, combined GPU reserve -640MiB.** После отдельного запроса пользователя default `GGML_SCHED_COPY_RESHAPE` включён ON; measured phase ниже использовал явные OFF/ON. Новый код и результаты не закоммичены; base my_maya9b74cc6e6/runtime6d038260e плюс [measured patch](artifacts/MEASURED.patch). Это продолжение согласованного [дизайна](../../designs/1017-hc-copy-sharing.md), не полный перенос Unsloth/Maya и не новое сравнение с ними.

## Реализация

Меняется только ggml/src/ggml-backend.cpp: protected pipeline slots/n_copies>1, CUDA->CUDA, direct full zero-offset contiguous F32 reshape от HC_POST. Canonical destination root уже должен быть input текущего split и иметь INPUT/OUTPUT protection. Его transfer/events/overwrite guards остаются. Только не-inplace RMS_NORM consumer получает destination view; allocation-only view вставляется перед первым consumer в graph_copy для корректного n_views. Нет глобального alias entry между split, ручной подмены data pointer, нового GPU buffer/kernel/stream, изменений allocator reuse policy или compute_splits.

Read-only prepass отклоняет roots с неизвестными/пишущими consumers, неподходящими views и input/export flags. Callback, CPU/другой backend, n_copies1, nested/offset/subrange/noncontiguous views и поздний/другой root используют старый путь. Dtype F32 задан валидным API HC_POST/HC_PRE; искусственный F16 HC_POST numerical test не создавался. [Протокол](artifacts/PROTOCOL.md), [измеренный source](artifacts/ggml-backend.after.cpp).

## Проверки точности и lifetime

| Gate | Результат |
| --- | --- |
| Малый CUDA0/CUDA1/CPU scheduler | 17 variants xOFF/ON/default =51 сочетание; exact outputs и правильный alias/fallback |
| Copy-slot rotation | 160 eligible chunks/mode, widths1/4/6/7/508/511/512/513; delayed host outputs exact |
| Число transfer requests | В measured opt-in этапе на160 chunks OFF/тогдашний default320, ON160; после promotion default160 |
| Capture/replay | По32 меняющихся input/mode; на каждой GPU ON capture1/launches31, update failures0; outputs exact к OFF |
| Compute failure/retry micro | По одной ошибке в каждом OFF/ON/default режиме, последующие outputs exact |
| Full target+draft | Два полных98K corpora,36 result groups OFF/ON/OFF; target/draft logits, seq/global/partial states и taps exact |
| Restore/append | Full/seq/partial checkpoint,90K resume,7676+512+4 tail; append1/7/511/513 exact |
| Speculation/rollback | Raw K3/K5 IDs/confidence/logits;36 zero/partial/full acceptance rollback workflows и следующий draft exact |
| Seed compute failure | Две контролируемые ошибки, очистка обеих state/range и full reload/retry exact |
| NLL text/code | По128 labels/corpus/mode: обе deltaNLL=0, PPL ratio1; raw logits/top1 exact |

Full helper: Q3_K-AntirezLike/MXFP8 Engram/Q8KV/BF16 DSpark,ctx100352/B8192/U512/P1/TopK2048/frozenallGPU/fit-cache-staging-mmprojOFF. K5/RS5/output6,p_min0/backend samplingOFF нужны для raw logits/confidence. M2/default LOCAL ON, ASYNC OFF. Observer подтверждает1836 shared RMS inputs только ON,0 OFF; M2 активен во всех tested paths. Native minimum free1325.25MiB>=512,exit0. Expected E-логи находятся только внутри двух FAULT_BEGIN/END; неожиданных ошибок нет.

Это transport-only gate относительно текущего runtime.128 labels на корпус не являются новой общей оценкой качества модели; согласованный предел +0.01nat/token выполнен точно. Source всех probes, raw states/logits hashes, fault results и library identity сохранены в [long summary](artifacts/long-summary.json), [raw result](artifacts/long-native/results.json), [micro result](artifacts/scheduler-counted-results.json).

## Clean native speed

Три отдельных server processes OFF-pre/ON/OFF-post, тот же ctx100352/layout/Q8KV, BF16K3/full/p_min0.5/backend samplingON, M2/default LOCAL, ASYNC OFF. Никаких preload/profiler/callbacks/calibration/builds в timed run. Full uncached raw input IDs,256greedy/seed1234/ignore_eos.512warm+2 measured,8Kwarm+3,98Kwarm+1 на каждом процессе. Всего27 запросов; полные output arrays и draft generated/accepted counters exact для каждого prefix во всех arms, включая warmup.

OFF ниже - среднее двух OFF-медиан, ON - медиана своего arm:

| Prefix | OFF PP,t/s | ON PP,t/s | DeltaPP | OFF TG,t/s | ON TG,t/s | DeltaTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 546.35 | 568.56 | +4.06% | 43.44 | 43.55 | +0.27% |
| 8192 | 1042.71 | 1087.14 | +4.26% | 48.45 | 48.63 | +0.37% |
| 98304 | 1066.43 | 1106.66 | +3.77% | 46.35 | 46.11 | -0.50% |

PP OFF drift между первым и последним control: -0.11% на512,-0.44% на8K,+0.19% на98K. Диапазоны PP8K OFF-pre1043.20..1046.61, ON1082.76..1088.52, OFF-post1040.11..1042.62. Это поддерживает полезный PP gain в данном fixture. Один порядок процессов, один speed corpus и singleton measured98K ограничивают перенос вывода. 98K TG delta-0.50% - одиночная точка; для вывода о небольшой TG регрессии нужна отдельная серия. Заметного TG выигрыша этот опыт не установил.

Это полный engine PP/TG, не оценка из суммы memcpy durations. Новый runtime не профилировался Nsight в timed runs. [Historical staged-copy screen](../1017-hc-copy-design-20261103/REPORT.md) остаётся отдельным свидетельством пути, не новым attribution всего4% только PCIe. [Все request/arm data и ranges](artifacts/bench-results.json).

## Память, размещение и завершение

Шесть свежих no_alloc cases:100352/default/OFF/ON,176128/ON,100352/K5OFF/ON. В измеренном этапе default/OFF budgets exact; model/KV и metadata per-tensor routes прежние,15360 routed GPU bundles, host routed bytes0. Это fixed-layout measure, не свободный auto-fit solver. Combined bound target+draft/taps reserve уменьшается по160MiB на CUDA1/2/3/4, суммарно640MiB. Standalone target component также уменьшается на CUDA5, но после binding taps37/38 этот выигрыш там исчезает;800MiB combined не заявляется.

В fresh ON production process NVML подтвердил +160MiB free на четырёх картах относительно OFF. Global sampled minimum почти прежний из-за другой bottleneck GPU: OFF-pre1435.25MiB, ON1439.25MiB, OFF-post1439.25MiB; все arms выше512. В K5 numerical process начальные OFF capacities удерживаются при переключении ON, поэтому он не является тестом уменьшения fresh ON capacity. Actual model buffers target и sidecar совпали во всех трёх arms; per-tensor parity проверена в metadata. Отдельного loader observer в clean timed processes не было.

NVML sampling100ms, stop guard200ms только для owned process; это sampled floor, не гарантия отсутствия пика между samples. Все процессы exit0, без OOM/CUDA error/compute-size mismatch, forced kill или floor breach. После опыта GPU compute apps отсутствуют. Actual176128 с новым flagON не запускался: там выполнен только no_alloc; прежний M2 actual176128 относится к copy-sharing OFF. [Подробности памяти](artifacts/summary.json), [provenance](artifacts/provenance.json).

## Решение и использование

Проверенный bounded путь включён по умолчанию по запросу пользователя: prefill gain полезен, точность и lifetime gates пройдены. Первоначальный opt-in после измерений по запросу пользователя переведён в default ON. Дополнительная переменная для включения не нужна; откат к исходным копиям:

```bash
GGML_SCHED_COPY_RESHAPE=0 ./llama-server ...
```

Unset или1 разрешает sharing при guards;0 возвращает исходные копии. Это runtime environment variable, не INI/CMake option. Пользовательские presets/my_build и GGUF не менялись. Нового commit/deploy/merge нет. Default promotion выполнен только после отдельного указания пользователя.

Другие кванты/LLM, F16KV, parallel>1/media/LoRA, cache/spill и произвольные view-sharing режимы этим набором не закрыты. Следующие отдельные пункты: dense idx_score/candidate_mask traffic, matched Maya и Q2/Q4 MMQ. Для нового расширения alias scope сначала нужен новый lifetime/graph design. [Артефакты и checksums](artifacts/SHA256.json).

## Promotion default ON

По отдельному запросу пользователя после full explicitON проверки изменён только unset-selector. Все dtype/backend/layout/split/lifetime/callback guards прежние. Новая сборка llama-server готова. Operator OFF/ON/unset повторён:51 сочетание, default и1 дают160 aliases/160 copy requests против320 при0, outputs exact. Fixed-graph capture32outputs/mode и31launches/GPU снова прошли. Fresh no_alloc default/1/0: default exact к1 по model/context/compute и routes;0 сохраняет baseline budgets.

Длительный target/draft/NLL/native OFF/ON/OFF не повторяли, поскольку explicitON вычислительный путь не менялся. [Selector delta](artifacts/promotion/DEFAULT_ON.patch), [полный promoted patch](artifacts/promotion/PROMOTED.patch), [новые проверки и hashes](artifacts/promotion/summary.json). Старые measured source/results/hashes сохранены отдельно.


## Последующее actual defaultON продолжение

В новом [этапе176128/indexer audit](../1017-hc-long-indexer-20261103/REPORT.md) выполнен свежий actualctx176128 с defaultON:172032+16 и повтор8K, sampled minimum761.25MiB,exact placement/IDs,exit0. Фраза no_alloc-only выше описывает исходную measured/promotion фазу; её raw artifacts и checksums не менялись.
