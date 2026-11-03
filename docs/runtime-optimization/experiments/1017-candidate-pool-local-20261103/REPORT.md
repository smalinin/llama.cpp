# DS41 producer-local candidate pooling, 2026-11-03

**Прототип реализован и проверен; флаг остаётся OFF по умолчанию.** Это продолжение [согласованного проекта](../../designs/1017-indexer-placement.md), текущая база my_maya`c3c91dda8` с HC-sharing defaultON. Начальный запрошенный коммит выполнен до этой работы; новые изменения этого этапа пока не закоммичены. User presets/my_build, GGUF, оригинальные веса и defaults не менялись.

## Что изменено

В deepseek41.cpp добавлены только callback names существующих CONT/padding/CONCAT и первого ADD после pooling. В graph_get_cb проверяется вся цепочка до назначения backends. Actual persistent index-key buffer должен принадлежать тому же CUDA device, что layer20, а fused score и вся подготовка должны поддерживаться этим backend. Score/CONT/FILL/CONCAT/POOL привязываются к нему; первый post-pool ADD остаётся у следующего query owner24, чтобы selector/dense mask не переехали вслед за pool и не создали другую большую копию.

Новые kernels/streams/буферы/форматы, scheduler lifetime/events/copy-slot policy и arithmetic operations не вводились. Block8, top2048blocks, independently final2048positions, causal masking, newest-block+inf pin, -inf padding и existing CUDA max/tie/nonfinite semantics прежние. Source20 имеет ratio1; это не замена block selector финальным source TopK.

Opt-in `LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1`. Unset/0 и иные значения OFF. Guards: deepseek41/source20/ratio1/block8/top2048/index_topk2048/next index-source24, CUDA layer split, supported fused LID/KV offload/causal, P1/text/one stream,6<ubatch<=512,без eval/abort callbacks и LoRA. Unsupported shape/backend/mode сохраняет обычный путь. Это bounded эксперимент данной Q3 конфигурации, не подтверждение всех вариантов LLM/media/parallel/KV.

## Размещение и резерв

Fresh no_alloc OFF/ON100352 и176128: score20 остаётсяCUDA2; CONT/poolCUDA3->CUDA2; first pin-ADD/selector/mask остаютсяCUDA3. Guards для widths<=6 дают те же24 unique placement tuples в каждом ctx. Все15360 bundles GPU, per-tensor routes/model/KV exact.

- ctx100352: combined no_alloc compute delta -593.24MiB; model/KV/routes exact.
- ctx176128: combined no_alloc compute delta -1073.04MiB; model/KV/routes exact.

В частности, расчётный резерв CUDA3 уменьшается на710.50/1247.00MiB, CUDA2 растёт на117.25/173.96MiB. Это combined metadata estimate; нельзя выдавать его за одинаковое увеличение global sampled minimum на каждой карте. [Metadata results](artifacts/metadata-results.json).

## Численные и lifecycle gates

- Mutable OFF/ON/OFF: два полных98K corpora,36 result groups,36 zero/partial/full rollback workflows,append1/7/511/513,seq/global/partial restore/resume. Target/draft raw logits, states, taps37/38/39, raw K3/K5/confidence и следующий draft exact.
- NLL128 labels/corpus/mode: text и code обе deltaNLL0, PPL ratio1. Это continuation diagnostic, не новая общая оценка PPL модели.
- По две controlled seed failures/full reload-retry в mutable и freshON фазах; обе state очищаются и recovery exact.
- Отдельный fresh constantON process с флагом уже при конструкторе: ещё12 full result groups двух corpora exact к OFF reference,minimum1325.25MiB,exit0без compute-size warning/OOM/CUDA error/forced kill.
- Corrected operator:80 boundary/pattern cases,query widths1/7/511/512/513,n_pos16384/16385/16391/16392,finite/all-inf/ties/NaN+inf; по2 input iterations и два **проверенных** GPU placements. Pool/top/mask bitwise exact. Отдельно по32 меняющихся inputs на свежих backend contexts; fixed-path capture/replay и отсутствие update failures проверены отдельно от shape-changing graph transitions.

### Две исправленные ошибки стенда и warning при смене флага

Mutable тест загружал OFF, затем включал ON в уже живом контексте. CUDA2 резерв вырос с expected1172.8325 до retained1197.9609MiB и при shutdown вывел ровно один size warning. Это исходный strict gate отверг. FreshON с самого конструктора сразу резервирует1206.26MiB и завершился без warning; все12 групп дополнительно совпали с OFF. Флаг следует задавать перед загрузкой нового контекста; динамическое переключение environment не является production-интерфейсом. Warning обычного сервера нельзя игнорировать на основании этого fixture.

Измеренный fresh runner после child exit0 ошибочно прочитал старый36-group JSON при ожидании12; raw manifest оставлен complete:false/error. Исправленный независимый validator проверил фактические command/output12groups, source identity, masks/state/rollback/faults/memory/clean shutdown. Логи и первоначальная ошибка не переписаны.

Первый operator run автоматически сбросил ручные backend hints при implicit allocation: обе ветки шли наCUDA0. Он **не принят** как two-GPU evidence. В operator-final.cpp добавлены explicit alloc_graph после назначения и assertions actual backend IDs. Corrected gate выполнен после native/memory серии без влияния на timings. Shape-changing graph update fallbacks в80-case серии не приравниваются к ошибке результата или fixed32 replay failure.

[Mutable validation](artifacts/long-summary.json), [fresh validation](artifacts/fresh-on-summary.json), [operator final](artifacts/operator-final-results.json), [fixed32 stats](artifacts/summary.json).

## Измеренный и финальный source

Raw precision gates использовали [MEASURED.patch](artifacts/MEASURED.patch). Перед native benchmark добавлена одна проверка `next_source < n_layer`: исчерпанный поиск у будущего24-layer checkpoint иначе мог совпасть с sentinel24. Текущая модель40layers/next24 идёт по прежней ветке. Final no_alloc повторил budgets/routes/owners exact; native и actual176128/copy proof используют final libraries. Это не arithmetic/default change. [Guard delta](artifacts/FINAL_GUARD.patch), [полный final patch](artifacts/FINAL.patch), [library/source hashes по фазам](artifacts/provenance.json). Precision hashes не выданы за новый binary run.

## Clean native OFF/ON/OFF

Q3_K-AntirezLike10shards/MXFP8Engram/Q8_0K/V/BF16 DSparkK3/full/p_min0.5/backend samplingON,ctx100352/B8192/U512/P1/TopK2048,прежний frozenallGPU split/overrides,fit/cache/staging/mmprojOFF. HC/M2/producer-LOCAL defaultON,ASYNCOFF; полный env включает прежние frozen admission overrides и хранится в manifest. Это не сравнение с Maya/Unsloth или универсальный preset.

Три отдельных процесса,27 full uncached requests по256greedy/seed1234/ignore_eos.512warm+2 measured,8Kwarm+3,98Kwarm+1 на каждом arm. Без LD_PRELOAD/profiler/callbacks/calibration/builds в timed runs. OFF ниже - среднее двух OFF-медиан, ON - его медиана.

| Prefix | OFF PP,t/s | ON PP,t/s | DeltaPP | OFF TG,t/s | ON TG,t/s | DeltaTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 566.47 | 565.30 | -0.21% | 43.47 | 43.15 | -0.75% |
| 8192 | 1083.87 | 1080.83 | -0.28% | 48.60 | 48.46 | -0.29% |
| 98304 | 1106.39 | 1122.53 | +1.46% | 46.23 | 45.73 | -1.07% |

Все27 output arrays и draft generated/accepted counters exact; actual model buffer logs одинаковы, все процессыexit0без allocation warning/forced kill/floor breach. По одному arm order и singleton measured98K ограничивают интерпретацию небольших deltas, особенно TG; ranges и OFF drift сохранены в [bench-results](artifacts/bench-results.json). Byte ratio8x не является8x engine gain.

## Actual176128 и pointer-copy proof

Свежий finalON ctx176128:8192+256,172032+16,8192+16,cache_n0,per-tensor placement exact к metadata,15360GPU bundles/host routed bytes0.8K и172K IDs exact к prior controls, повторный8K16 к first16. Minimum761.25MiB>=512,exit0без size mismatch/OOM/CUDA errors/fallback/forced kill.

| Logical GPU | Inference min free, MiB |
| --- | ---: |
| CUDA0 | 1139.25 |
| CUDA1 | 1995.25 |
| CUDA2 | 2005.25 |
| CUDA3 | 2017.25 |
| CUDA4 | 761.25 |
| CUDA5 | 1143.38 |

NVML100ms/owned-process guard200ms: sampled floor, не гарантия отсутствия пика между samples. Допуск рассчитан по measured per-device overhead+64MiB, не только no_alloc. Уменьшать ctx не потребовалось; auto-fit262K этим не закрыт.

Это отдельный диагностический/memory процесс, не speed A/B. Loader observer и pointer probe читают CPU metadata, не tensor payload; нет GPU probe buffers, eval callback/backend iface mutation, дополнительных CUDA sync. Probe armed только для172K request. Он требует exact full src+dst pointers/bytes/ne/nb/type,protected destination clone,expectedCUDA ordinals,actual consumer и stable metadata; partial/address-only совпадения исключаются.

- Prefill widths>6: full idx_score20 CUDA2->3 requests: **0**.
- Pooled block scores CUDA2->3: **134**, logical payload **3.338GB**.
- Dense candidate mask CUDA3->4: **134**, logical payload **26.700GB**, ровно8xblock bytes в данном aligned fixture.

В short tail/verification widths3/4 записаны ещё5 небольших full-score и5mask requests (13.09MB каждого): это предусмотренный fallback, не wide prefill. Probe охватывал весь172K request; classification использует canonical query width, а не ne1 произвольного reshape. Копии block scores достигают прежнего pin-ADD; no new full reverse score edge. Это логические CUDA API requests, не hardware DMA counters. Поздний dense mask transfer остаётся отдельной задачей. [Full pointer analysis](artifacts/audit-analysis.json), [actual payload](artifacts/payload-176128/results.json).

## Статус и следующий шаг

Прототип оставлен opt-in/OFF до review. Для опытного запуска в проверенном режиме:

```bash
LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 ./llama-server ...
```

Следующий отдельный кандидат - передавать block keep и разворачивать dense mask локально у actual consumers. Для него нужен собственный ownership/lifetime design; он не включён в этот patch. Matched Maya/Q2/Q4 и остальные режимы остаются в плане.

[Артефакты, replay notes и raw ошибки](artifacts/README.md), [checksums](artifacts/SHA256.json). Пользовательские model files/presets/my_build не изменены, нового commit/default promotion/deploy после initialc3c91dda8 нет.
