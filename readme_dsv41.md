# DeepSeek-V4.1-Flash-1017: использование новых возможностей

Для ветки **`my_maya`**, начиная с `e4d423309`. В `my_build` эти последние изменения ещё не перенесены. Бинарник текущей сборки: `/home/sergei/Github/llama.cpp/build-aliceai/bin/llama-server`.

## Кеш экспертов и автоматический выбор CPU/GPU при prefill

```bash
MODEL=/path/to/model-00001-of-00010.gguf
SERVER=/home/sergei/Github/llama.cpp/build-aliceai/bin/llama-server

LLAMA_MOE_PREFILL_STAGE=auto "$SERVER" \
  --model "$MODEL" \
  --batch-size 2048 --ubatch-size 512 \
  --ctx-size 256000 --parallel 1 --threads 20 --threads-batch 20 \
  --flash-attn on --load-mode mmap --lazy-mode auto \
  --override-kv deepseek41.attention.indexer.top_k=int:2048 \
  --moe-expert-cache 64 \
  --moe-expert-cache-profile "$MODEL.moe-routes.json" \
  --jinja --host 0.0.0.0
```

Сохраните своё размещение весов/GPU и настройки sampling. `CUDA_VISIBLE_DEVICES`, если используется, задаётся перед этой командой.

- `--moe-expert-cache 64`: до 64 GPU-слотов на каждый подходящий слой с host-resident экспертами; кешируются его host-проекции. Полностью GPU-resident слои такой кеш не используют. Без параметра кеш выключен.
- `LLAMA_MOE_PREFILL_STAGE=auto`: при загрузке измеряет CPU и существующий pinned GPU staging для host Q3_K-проекций DeepSeek41. Выбирает GPU только при выигрыше минимум 5% на каждом проверяемом слое группы и достаточной VRAM с резервом 1024 MiB; иначе явно выполняет host-проекции на CPU. Нужен включённый expert cache.
- Auto рассчитан на полный chunk **512**; короткие prefill-хвосты 33–511 идут через CPU. Decode/verification до 32 токенов сохраняют свой кеш-путь. `ubatch-size=8192` для этого режима не задаём.
- При первом запуске калибровка увеличивает время загрузки. Результат сохраняется рядом с первым шардом: **`<first-shard>.gguf.moe-prefill-costs.json`**. При следующем запуске выполняется короткая перепроверка; изменения модели, сборки, CPU/GPU или drift вызывают новый замер. Если каталог недоступен для записи, costs остаются в памяти с сообщением в логе.
- `LLAMA_MOE_PREFILL_STAGE_COSTS=/path/costs.json` задаёт другой путь. Для честного auto не задавайте `LLAMA_MOE_PREFILL_STAGE_DEVICE`: это ручной override. `LLAMA_MOE_PREFILL_STAGE=0` отключает staging/auto и возвращает прежний scheduler.

На проверенной Q3 auto выбрал CPU: PP вырос с 251.79 до 365.22 t/s при 512 и с 246.77 до 347.58 t/s при 8K; TG практически не изменился. Это результат конкретного размещения со spill на CPU, не обещание такого выигрыша для любого кванта.

## Лимит VRAM и сохраняемый профиль экспертов

Вместо `--moe-expert-cache 64` можно задать общий бюджет на одной карте:

```bash
--moe-expert-cache-mib 4096 --moe-expert-cache-device CUDA5
```

Это альтернатива slots-режиму: одновременно их не задавать. `CUDA5` — логический номер после `CUDA_VISIBLE_DEVICES`; выбирайте существующий GPU-home нужных слоёв. Параметр не переносит вычисления слоя на другую карту. Бюджет включает кеш и staging на этой же карте, но не весь KV/compute footprint сервера. Межслойные partitions пока фиксированные.

`--moe-expert-cache-profile FILE` читает статистику маршрутов при загрузке и атомарно сохраняет её при штатной выгрузке. Если файла нет, начинает с пустого профиля; каталог должен существовать. В `llama-server` профиль фиксируется одним snapshot до `fit`; metadata-пробы и реальная загрузка используют одинаковые исходные counts, а samples весов проверяются при реальной загрузке. Это **другой файл**, не costs-калибровка. Профиль покрывает наблюдаемые host-cache слои, а не все маршруты модели.

Опционально `LLAMA_MOE_CACHE_PRELOAD=1` загружает популярные эксперты из этого профиля до первого запроса. По умолчанию preload выключен; без накопленной статистики/seed прогревать нечего. Сырые файлы Maya напрямую в параметр profile не подставляются. `--moe-expert-cache-inserts 2` — текущий default лимита новых uploads на слой за decode step.

Для экспериментального распределения заданного MiB-cap по профилю добавьте перед запуском:

```bash
LLAMA_MOE_CACHE_CAPACITY_PROFILE=1 LLAMA_MOE_PREFILL_STAGE=auto ./llama-server ... \
  --moe-expert-cache-mib 4096 --moe-expert-cache-device CUDA5 \
  --moe-expert-cache-profile /path/target-routes.json
```

Флаг по умолчанию OFF. Минимум один слот на подходящий слой; остаток распределяется по доле маршрутов на байт со сглаживанием 10% к равномерному prior. Пустой профиль даёт равномерный fallback. Slots закреплены за слоями; LRU прежний. Сам общий бюджет VRAM этот режим не выбирает. Legacy `--moe-expert-cache 64` сохраняет прежнее поведение.

## Замена через резервный слот

По умолчанию spare-замена сохраняет старого resident-эксперта доступным, пока новый копируется в отдельный spare. Готовый bundle публикуется между запусками графа; текущий LRU victim освобождает место для следующей загрузки. Zero dummy остаётся отдельным и не перезаписывается. Ошибка upload не удаляет старый mapping.

Spare-замена включена по умолчанию при включённом expert cache; `LLAMA_MOE_CACHE_SPARE=0` до запуска процесса возвращает прежнюю замену без spare. Работает с slots-режимом либо MiB-cap, в том числе с профильными ёмкостями. Сам expert cache по-прежнему включается отдельным CLI-параметром. При `--moe-expert-cache 64` остаётся до 64 полезных экспертов, но добавляется один физический spare на слой. При фиксированном MiB-cap spare входит в бюджет и уменьшает полезную ёмкость; сравнивайте одинаковые bytes. Общий бюджет этот флаг не выбирает; политика вытеснения задаётся отдельно, как описано ниже. В коротком Q3/BF16 sidecar A/B при 900 MiB TG geomean изменился на +1.43%; значимого выигрыша не подтверждено.

При включённом expert cache и spare политика `heat` используется по умолчанию. Для возврата к LRU задайте `LLAMA_MOE_CACHE_POLICY=lru` до запуска. При `LLAMA_MOE_CACHE_SPARE=0` или `off` без явной policy выбирается прежний LRU без spare; явное сочетание `LLAMA_MOE_CACHE_POLICY=heat` с отключённым spare отклоняется.

Heat растёт по исполненным decode/verification rows до32 токенов, включая отклонённые draft-позиции. Каждые256 наблюдаемых токенов слоя heat умножается на0.9. Входящий эксперт должен пройти строгий порог `heat_new > 1.25 * heat_victim + 1` при постановке upload и перед публикацией; между равными по heat resident-экспертами выбирается LRU. Сохранённый route profile, если задан, нормируется к суммарному стартовому heat512 на слой. Prefill онлайн heat не обновляет. Это эвристика, не полный перенос политики Maya и не гарантия ускорения. Короткий Q3/BF16 A/B512/8K: TG geomean +1.80%; по последующей команде пользователя heat включён по умолчанию.

## DSpark sidecar

К основной команде можно добавить наш BF16/MXFP4 sidecar или совместимый BF16 Unsloth:

```bash
--spec-draft-model /path/to/dspark-model.gguf \
--spec-type draft-dspark \
--spec-draft-n-max 3 \
--spec-draft-p-min 0.5 \
--spec-verify-policy full \
--gpu-layers-draft all
```

`full` — политика по умолчанию; confidence floor может уменьшить фактически проверяемый draft. Начните с `n-max=2` или `3`, затем сравните с `5`: большее значение не гарантирует ускорения. KV packing поддерживает до шести target rows — anchor плюс пять draft-токенов; отдельные matmul/MoE guards сохраняются. DSpark не является обычным `draft-mtp`.

Для экспериментальной **adaptive** замените соответствующие параметры:

```ini
spec-verify-policy = adaptive
spec-draft-n-max = 3
spec-draft-p-min = 0
parallel = 1
```

Оставьте `spec-type = draft-dspark`; `spec-draft-n-min` должен быть не больше 1. Adaptive выбирает K=0..3 до draft forward. `spec-verify-costs-us` необязателен: полные cycle costs измеряются во время начальных запросов и обновляются online. Это отдельный механизм от сохраняемой MoE startup-калибровки.

Опциональные admission-эксперименты, оба по умолчанию выключены: `LLAMA_DSPARK_CUM_P_MIN=0.5` ограничивает произведение confidence префикса, `LLAMA_DSPARK_EMPTY_COOLDOWN=1` пропускает следующую попытку после пустого admission. Для исходного сравнения оставляйте их выключенными.

## Что уже включено по умолчанию

Эти переменные обычно задавать не требуется; fast paths работают только при соблюдении своих guards.

| Переменная | Назначение |
| --- | --- |
| `LLAMA_DSV41_VERIFY_BATCH` | Короткая batched target verification |
| `LLAMA_DSV41_VERIFY_PACK` | Упаковка KV для подходящего batched-пути |
| `LLAMA_DSV41_VERIFY_KV_WIDE` | Расширенная упаковка до anchor + 5 drafts |
| `LLAMA_DSV41_VERIFY_FLOAT_BATCH` | Floating projections через независимые samples с сохранением single-column reduction |
| `LLAMA_DSV41_VERIFY_QUANT_SAMPLES` | Аналогичная упаковка quantized dense projections при 5–6 target rows |
| `GGML_CUDA_MMVF_SAMPLE_BATCH` | CUDA floating matvec для подходящих небольших sample batches |
| `LLAMA_DSV41_PREFILL_COMPACT` | Compact prefill при подходящем режиме и пороге 24576 compressed-записей |
| `GGML_SCHED_COPY_RESHAPE` | Общая CUDA-копия HC root/reshape в подходящем pipeline |
| `LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL` | Компактная передача block keep и локальное разворачивание candidate mask |

Для диагностики соответствующий путь отключается значением `0` **до запуска процесса**. При `--parallel > 1`, неподходящем KV-типе или другой геометрии часть verification-оптимизаций использует fallback. Для проверки последнего пути сохраняйте `--parallel 1`, flash attention и F16 KV.

## Диагностика

`LLAMA_DSPARK_PROFILE=1` включает подробную телеметрию draft/verification/rollback; для замеров обычной скорости оставляйте выключенным, поскольку profiling может синхронизировать графы. Переменные окружения задавайте перед запуском родительского router-сервера с `--models-preset`; обычные CLI-опции в preset записываются без `--`.

Сравнивайте одинаковые модель, sidecar, prefix, index_topk, sampling и размещение. Для sidecar сначала 512 и 8K; к 25K/60K переходите только при значимом ускорении первого этапа.

Подробности реализации: [DeepSeek-V4.1](docs/deepseek-v4.1.md) и [калибратор MoE](tools/moe-cache-calibrate/README.md). Отчёты и планы экспериментов находятся в отдельном репозитории документации `_my_deepseek41`, в `validation/moe-cache/` и `DEEPSEEK_V41_1017_MAYA_EXPERT_CACHE_PLAN.md`.

## Экспериментальный pipeline с GPU-only overrides

Три opt-in флага, все по умолчанию выключены:

```bash
LLAMA_PIPELINE_GPU_OVERRIDES=1 LLAMA_PIPELINE_PREFILL=1 \
GGML_CUDA_ARGSORT_RADIX=1 "$SERVER" \
  --model "$MODEL" \
  --fit on --fit-target 1024 \
  --ctx-size 128000 --batch-size 8192 --ubatch-size 512 --parallel 1 \
  --threads 20 --threads-batch 20 \
  --cache-type-k f16 --cache-type-v f16 --flash-attn on \
  --load-mode mmap --lazy-mode auto \
  --override-kv deepseek41.attention.indexer.top_k=int:2048
```

`LLAMA_PIPELINE_GPU_OVERRIDES=1` разрешает существующий pipeline при совместимых фактически применённых overrides между активными GPU. Проверяются target buffer и реальное размещение; CPU/routed-host эксперты, в том числе combined gate/up, остаются на прежнем пути. Неиспользуемое правило для отсутствующего тензора само по себе не считается CPU-spill. Остальные условия pipeline, включая async/events, сохраняются. Это относится и к overrides, которые создаёт auto-fit без ручного CLI-флага.

`LLAMA_PIPELINE_PREFILL=1` выделяет четыре отдельные копии входных буферов и owned host staging для асинхронных передач. Перед записью ожидается завершение чтения текущей копии. Входы отделены от compute scratch: смена плана размещения без роста буферов не требует общего ожидания всех GPU. Физическое увеличение буферов, ошибки вычисления, чтение результатов и teardown сохраняют синхронизацию. Для multi-token prefill граф пересобирается, чтобы продвигать копии; прежний TG reuse сохраняется. Путь ограничен DeepSeek41 default graph, одной последовательностью, без nextn extraction, eval и abort callbacks; остальные режимы сохраняют обычный путь. DSpark target использует layer-input taps, поэтому guard по nextn сам по себе sidecar не исключает. Отдельно проверена Q3/BF16 DSpark конфигурация с taps37..39, K5/p-min0.5, ctx128000 и draft devices CUDA5,CUDA0; это не общий допуск nextn, multimodal или parallel>1. Сам sidecar подключается отдельными spec-параметрами.

`GGML_CUDA_ARGSORT_RADIX=1` выбирает существующий `DeviceSegmentedRadixSort` и вне CUDA graph capture. Он избегает чтения служебных счётчиков CUB на CPU, которое мешает enqueue нескольких чанков. При выключенном флаге сохраняется прежний выбор сортировки; capture уже использовал radix раньше. Для равных scores порядок индексов в общем случае может зависеть от алгоритма, поэтому корректность Top-K/logits проверяется отдельно. Это экспериментальный флаг, а не изменение типов весов или KV.

`LLAMA_DSV41_VERIFY_REUSE` включён по умолчанию для описанного ниже совместимого async pipeline пути. `LLAMA_DSV41_VERIFY_REUSE=0` отключает reuse verification. При async pipeline разрешает прежний `can_reuse` для verification2..6 токенов в пределах n_rs_seq+1, когда запрошены все logits. Scheduler fence перед записью входов сохраняется; обычный prefill продолжает вращать четыре копии. На проверенной Q3/BF16 конфигурации text/code512/8K logits, taps и partial rollback совпали точно. Диагностический TG gain около1–3%; независимый native speed gate ещё не закрыт.

`GGML_CUDA_MMVQ_SAMPLE_BATCH` по умолчанию включает дополнительный Q8_0 CUDA путь для sample batches5/6 на Ampere/Ada. `GGML_CUDA_MMVQ_SAMPLE_BATCH=0` отключает его. Он сохраняет single-query редукцию и объединяет обработку запросов в одном block. В DeepSeek verification используется вместе с `LLAMA_DSV41_VERIFY_QUANT_SAMPLES` (уже ON по умолчанию). Small-K, другие widths/types, fusion и неподдержанные устройства сохраняют прежний путь. На Q3/BF16 DSpark fullK5/p-min0.5, ctx128000, B8192/U512 native A/B дал +1.7..8.6% TG на512/8K; logits/taps/state/rollback совпали точно. Prefill ускорения эта правка не обещает.

`GGML_CUDA_MMQ_MOE_TILE` по умолчанию использует тайл32 для routed Q3_K MMQ prefill на Ampere/Ada. `GGML_CUDA_MMQ_MOE_TILE=0` возвращает прежний выбор. Ограничения: >=128 экспертов, chunk64..512 и достаточно большой grid для сохранения whole-K редукции. Остальные формы используют прежний путь. Задавайте переменную до запуска процесса. На Q3, ctx128000, B8192/U512, topk2048, без sidecar native A/B показал примерно +26..30% PP на512/8K при точном совпадении logits и KV-состояния; TG практически не изменился. Отдельный native A/B с BF16 DSpark fullK5/p-min0.5 подтвердил +24..28% PP на512/8K;600 позиций logits/taps/KV и30 partial rollback/resume совпали точно. TG fresh практически прежний; cached wiki/code изменился на+0.97%/-1.18% в коротком контроле. При том же U512 logical batch8192 дал на8K примерно на11% больше PP, чем2048. Parallel>1 не проверялся.

Pipeline увеличивает workspace. Auto-fit учитывает выбранный режим при metadata estimate. На проверяемой Q3 при context256000 не получилась нужная полностью GPU раскладка с pipeline; для эксперимента использован context128000 и reserve1024MiB. Значение128000 не является универсальным пределом. Не фиксируйте одновременно старый `tensor-split`/`gpu-layers`, если хотите дать fit перераспределить память. Смотрите фактические eligibility/pipeline/fallback сообщения, а не только наличие флагов.

Engram остаётся на обычном CPU/lazy пути. Квантование, sampler и математическая семантика attention не меняются. Эти флаги не являются общим автоматическим VRAM allocator и не означают поддержку pipeline с изменяемым CPU expert cache.

CUDA graph cache можно увеличить вручную, задав перед запуском `GGML_CUDA_GRAPH_CACHE_CAPACITY=128` и при необходимости `GGML_CUDA_GRAPH_CACHE_TTL_MS=30000`. Defaults остаются64 entries на backend context и10000ms; диапазоны1..512 и1..60000, некорректное значение возвращает default. TTL проверяется при sweep каждые5s. Переменные задаются до запуска процесса. На Q3/BF16 DSpark, ctx128000, B8192/U512, topk2048:128/10s дал около+1.9% TG,256/10s+2.6%;256/30s около+3.7% против64/10s. PP практически прежний. Это короткий последовательный A/B, не универсальная гарантия. У256/10s process RAM вырос примерно на2.15GiB, сумма пиков VRAM по картам на990MiB; entry cap не является byte-budget. При тесном размещении экспертов оставьте64. Keys, warmup, property checks и численная семантика не меняются.

## CPU-декодирование MXFP8 Engram

LUT decoder включён по умолчанию для CPU MXFP8: таблица256 F32 значений заменяет повторные `ldexpf`, при прежнем E8M0 scale и scalar fallback для exceptional blocks. Отключить перед запуском: `GGML_CPU_MXFP8_LUT=0 ./llama-server ...`; `=1` явно включает LUT. Другие значения отключают его. Q5_K/Q8_0 Engram и CUDA decoder не менялись.

Побитовые CPU-проверки (включая rounding/FTZ/NaN/zero), logits/KV/taps/rollback полной Q3 и короткие native greedy пары совпали. На Q3 с MXFP8 Engram наблюдалось +4.24% PP512, +0.82% PP8K text/+0.49% code; TG менее1%. Это не обещание ускорения длинного prefill: CPU decoder665.9 ->170.4 ms на8K не стал равным wall gain. Новых payload buffers/threads нет, таблица занимает1 KiB.


## Q8_0 KV: текущий статус

Q8_0 KV пока экспериментален; F16 остаётся по умолчанию. При явном
`--cache-type-k q8_0 --cache-type-v q8_0` поддерживаются локальные LID replicas
на подходящем single-sequence CUDA layer split; `LLAMA_DSV41_LID_REPLICAS=0`
возвращает shared-source вариант. Guards по размещению, VRAM и fallback сохранены.
На Q3512/8K OFF/ON logits, taps, canonical state и rollback совпали точно.
Q8 indexed backend поддерживает counts на NVIDIA Ampere/Ada; encoded Q8 row-copy API (CPU/CUDA) и связка packing -> attention проверены побитово. Native compact graph Q8 shared64/MMA512 включён по умолчанию при выборе Q8_0 KV и совместимой конфигурации; ограниченные quality/lifecycle/memcheck проверки пройдены. Область проверки описана ниже.
`LLAMA_DSV41_Q8_PACK` сохраняет encoded Q8 при raw ordering
и короткой verification packing (2–6 tokens). По умолчанию ON. Работает в
ограниченном single-sequence CUDA layer split; causal masks и learned sinks сохранены.
При прежнем masked attention same-Q8 logits/state/taps/rollback совпали на1200 позициях512/8K.
Этот fallback распаковывает KV в F16 внутри CUDA.
При Q8 KV CLI выше оба env `LLAMA_DSV41_Q8_PACK` и `GGML_CUDA_DSV41_Q8_MMA`
включены по умолчанию; задавать их не требуется. Для отключения только native kernel:
`GGML_CUDA_DSV41_Q8_MMA=0`. Для прежнего пути без encoded packing:
`LLAMA_DSV41_Q8_PACK=0 GGML_CUDA_DSV41_Q8_MMA=0`.
На Ampere/Ada при одной последовательности/layer split, Top-K2048 и пороге24576
compressed-записей включается shared K=V MMA512/GQA64 с sinks/counts: decode,
prefill до512 и verification2-6. Неподходящие формы сохраняют прежний путь.
Kernel распаковывает только bounded tile; полный encoded Q8 concat пока остаётся.
`GGML_CUDA_DSV41_Q8_MMA_TRACE=1` и verbosity4 показывают реальный выбор kernel.
Grouping меняет округления: между masked и compact logits не bit exact.
Первый same-Q8 common-prefix98K screen:257 labels/corpus, Delta NLL text-0.000743 /
code-0.000101. Последующий расширенный full-prefix F16/Q8 masked/Q8 compact gate:
4096 scored labels на corpus/history/mode, fresh8192 и prefix98304+suffix8192.
Q8 compact относительно F16: text-0.000255/-0.002485, code+0.000294/-0.000468
(fresh/long); все случаи ниже диагностического ориентира+0.01 nat/token.
Это ограниченный technical prose/C++ набор, не общая оценка качества генерации.
На границе padded compressed KV исправлен выбор attention при partial rollback:
indexed Q8 использует общий padded pool с отдельными causal masks/counts.
На сохранённом106496-token prefix repeat/serial/2+2/rollback exact;
16 разрешённых partial rollback widths2-6 прошли,6/keep0 использует full checkpoint.
Стартовый резерв DeepSeek41 single-sequence теперь учитывает полный compressed KV
и короткие verification2-6 с taps/RS, а также partial/zero-output prompt chunks.
Metadata sweep F16/Q8 при contexts97280/100352/128000 не превышает startup reserve;
реальный Q84->2 контроль с output limit6 не увеличил compute buffers.
Paired fit-on проверен для Q3+BF16 DSpark fullK5/p-min0.5 при ctx128000,
`--fit on --fit-target 512`, исходной GPU-only раскладке target и draft CUDA5/CUDA0.
Prompt106496/вывод256 завершены без OOM, минимум свободной VRAM657MiB.
Это одна конфигурация; fit-target задаёт запас в оценке, а не строгий runtime byte-cap.
Добавлен startup reserve DFlash feature-injection path: для этого sidecar30MiB
входных features x4 scheduler copies =120MiB CPU_Host. После fix32 metadata формы
укладываются в reserve, реальный512/8K контроль сохраняет точные tokens/draft counts
и не выдаёт предупреждение о CPU_Host compute buffer.
Native same-Q8 matched A/B после98K подтверждает PP+12.1..12.4%, forced TG+16.4..17.4%.
Bounded task gate после общего98365-token reference: F16/Q8 masked/Q8 compact дают
по354/354 Python functional checks; общий запрещённый импорт heapq сохранён как
instruction FAIL. JSON, Russian facts и ручная factual проверка summary пройдены.
SVG во всех трёх упёрся в cap2048 и остаётся INCONCLUSIVE.
Compact warm/cancel-resume/erase-replay дали exact tokens; cancel сравнивался при
одинаковом cache boundary. Native whole-target memcheck на восстановленном106496
prefix:0 errors, rollback/replay exact, prefill32/512 finite, compute sizes прежние.
Parallel2 сохраняет fallback:8 metadata guard форм и4 реальных concurrent JSON
на ctx32768 пройдены; native compact для parallel>1 не включён.
Эти проверки не заменяют общий quality gate, завершённый SVG, instrumented prefix
с0 или допуск других placements/backbones. С2026-11-02 compact Q8 включён по
умолчанию для совместимого Q8_0 KV; guards и возможность отключения сохранены.
Подробности: отчёты `1017-q8-compact-native-20261101`, `1017-q8-quality-workspace-20261101`,
`1017-q8-paired-fit512-20261101` и `1017-q8-task-lifecycle-20261102`.
Прямой F16/Q8 screen с BF16 DSpark (fullK5/p-min0.5, одинаковое размещение,
ctx100352/fit512, один technical-prose fixture, warmup+3) показал tradeoff:
полный98K PP632.93->669.97t/s (+5.85%), TG34.08->28.07t/s (-17.63%).
На512/8K TG Q8 ниже на20.44%/7.86%; на98K output tokens совпали,
acceptance76.50%/76.89%. Это другое сравнение, чем same-Q8 kernel A/B выше.
Q8 даёт дополнительно566..710MiB sampled free VRAM на GPU; его минимум979MiB.
F16 min363MiB ниже fit-target512 без OOM/compute warning: строгий runtime
запас512 этим F16 запуском не подтверждён; источник estimate/peak расхождения
ещё не локализован. Результаты ограничены этим fixture и fixed F16->Q8 order.
Подробности: `1017-q8-f16-paired-ab-20261102`. Q8 не заменяет F16 по default.
`GGML_CUDA_Q8_CONCAT_VECTOR` включён по умолчанию для aligned
uint4 copy encoded Q8 concat: dim2,512 columns/one head,1..6 streams;
неподходящие alignment/strides/types/shapes сохраняют byte path. Для отключения
задайте `GGML_CUDA_Q8_CONCAT_VECTOR=0` до запуска процесса: CUDA graph может
сохранять выбранный при capture kernel.
Target-only Q3 ABA0/1/0 после106496: verification calls1..6 latency ниже
на10.85..14.61%, logits/state и4->2/6->1 rollback exact, buffers прежние.
Prefill512 без значимого изменения. Ada/Ampere38 operator cases на устройство,
operator memcheck0. Это target-only screen, не новый формат KV. Отдельный paired BF16 server gate
завершён: vector concat дал +3.06% TG на8K и +12.50% на98K,
при одинаковых output tokens/acceptance и без изменения памяти; PP прежний.
Полный compressed-pool repeat/concat сохраняется. Подробности:
`1017-q8-verify-profile-20261102`, `1017-q8-concat-vector-20261102`.
Общий F16 tile64 подключён отдельно через `GGML_CUDA_DSV41_F16_PREFILL64=1` (default OFF):
только compact F16 K=V, полный chunk512, single stream, head512/GQA64, sinks/counts,
Ampere/Ada. Порог24576 compressed-записей сохраняется; короткий decode/verification
использует прежний путь. `GGML_CUDA_DSV41_F16_PREFILL64_TRACE=1` и `--verbosity4`
показывают выбор kernel. Расширенный technical prose/C++ screen:4096 scored labels/corpus,
неповторяющийся prefix98304 +suffix8192. Полный prefix ON дал ΔNLL text−0,00407 /
code+0,00069; replay/rollback PASS. Native tail PP8K +9,8–10,2%, forced TG прежний;
fresh512/8K в пределах0,6%, хеши logits совпали. BF16 sidecar прошёл отдельный98K
smoke при ctx100352 и сбалансированном draft; исходная128K раскладка упиралась в VRAM.
Подробности: отчёт `1017-f16-prefill64-quality-20261101`. Logits/taps не bit exact;
default OFF сохранён до review и более широкого quality gate. Расширенный Q8 technical
NLL, bounded task/lifecycle/memcheck и target workspace gates описаны выше;
более широкий quality gate остаётся в плане; ограниченный target+draft fit512 gate пройден.
Прежний короткий code NLL screen дал+0.033 nat/token против F16 при ориентире+0.01.
Новый расширенный набор прошёл ориентир, но не повторяет все labels старого screen;
рекомендовать Q8 вместо F16 по умолчанию пока рано.


`LLAMA_DSV41_VERIFY_DUAL_KV` по умолчанию включает compact Q8 verification
с чтением двух KV-буферов: ordered raw отдельно для каждого запроса, compressed
pool общий для всех verification streams. Он убирает полный shared repeat/concat,
сохраняя существующий MMA, counts, sinks и порядок строк. Для отключения
задайте `LLAMA_DSV41_VERIFY_DUAL_KV=0` до запуска процесса; `=1` явно включает режим. Нужны `--cache-type-k q8_0` и
`--cache-type-v q8_0`, NVIDIA Ampere/Ada, layer split, parallel1 и короткая
verification на2..6 токенов. Порог24576 compressed-записей прежний.
Decode1, prefill, F16 и неподходящие режимы используют прежний путь.

Operator cases на Ada/Ampere побитово совпали с физическим concat, memcheck0.
Native OFF/ON/OFF: logits/state, save/restore и4->2/6->1 rollback exact;
latency короткой verification ниже примерно на29..37%. Full Q3+Q8KV+BF16 DSpark
при ctx100352, B8192/U512, TopK2048, fullK5/p-min0.5: TG98K
31.65->41.57t/s (+31.33%), на8K практически прежний.
Output tokens и draft counters совпали, PP прежний. Это bounded greedy fixture,
не общее сравнение с Maya или полный quality gate.

Первый server screen выявил недооценку compute workspace. Исправленный startup
и no_alloc/auto-fit консервативно резервируют packed fallback для промежуточных
форм; поэтому экономия выделенного workspace не заявляется. Финальный полный
server повтор завершился без allocation mismatch. Подробности и сохранённая
неудачная попытка: `1017-q8-dual-buffer-20261102`.


`LLAMA_DSV41_DECODE_DUAL_KV` по умолчанию включает чтение двух
KV-буферов для одиночного compact Q8 decode без sidecar. `=1` явно включает
режим, `=0` возвращает packed fallback; задавайте переменную до запуска процесса.
Условия те же: Q8_0 K/V, Ampere/Ada, D512/GQA64, одна sequence/layer split
и минимум24576 compressed-записей. F16 и prefill сохраняют прежний путь.
Operator/native проверки дали exact logits/state/rollback. Target-only server TG
на98K вырос с33.70 до37.43t/s (+11.04%); на8K скорость практически прежняя.
Подробности: `1017-q8-decode-dual-20261102`.

Недооценка CUDA2 workspace на18.77MiB исправлена дополнительным reserve
промежуточных логических состояний CSA/HCA. Физическая ёмкость KV views и kernels
прежние; no_alloc/auto-fit учитывают тот же максимум. На ctx100352 дополнительный
расчёт занимает около3.5s. Два свежих сервера без sidecar (decode OFF/default ON),
полное чтение8K/98K и16 greedy outputs на запрос: токены прежние, exit0 без
workspace mismatch, minimum free VRAM1571MiB. Это bounded lifecycle gate для
этой раскладки; другие context/ubatch/placements проверяются отдельно.
Подробности: `1017-q8-workspace-reserve-20261102`.

### GPU prefix masks (default ON)

`LLAMA_DSV41_GPU_PREFIX_MASK=1` строит CSA/HCA masks на CUDA устройствах
слоёв из небольшого массива `n_visible`, сохраняя значения `0/-Infinity`.
Это убирает передачу больших masks с CPU. Default ON: unset или `=1`
включает проверенный путь; `=0` возвращает прежний host mask.

Режим ограничен DeepSeek41, одной последовательностью, текстовыми
ubatch7..512, causal/offload KQV и layer split на поддерживаемых CUDA
устройствах. Чанки1..6 (включая prefill tails и verification) и неподходящие режимы
используют fallback.
F16/F32 masks проверены как operator; full-model gate относится к Q3 с
Q8_0 KV и BF16 DSpark. KV precision этим флагом не меняется.

Operator/native checks: exact masks, logits/state/replay и rollback;
CUDA memcheck на Ada/Ampere без ошибок. Full-prefix PP/TG и память:
отчёт `1017-gpu-prefix-mask-20261102` в validation/moe-cache проекта.


## GPU features для DSpark: producer-local по умолчанию

При подходящей конфигурации DSpark автоматически сохраняет три последних target taps в F32 GPU-буферах рядом с producer и собирает выбранный chunk на GPU encoder FC. Дополнительные переменные окружения для включения не нужны.

| Переменная окружения | Default | Назначение |
| --- | --- | --- |
| `LLAMA_DSPARK_GPU_FEATURES` | `1` | `0` возвращает прежний CPU feature-путь |
| `LLAMA_DSPARK_GPU_FEATURES_LOCAL` | `1` | `0` выбирает central GPU bridge при включённых GPU_FEATURES |
| `LLAMA_DSPARK_GPU_FEATURES_ASYNC` | `0` | `1` включает экспериментальный async ingress |

Устанавливать переменные до запуска llama-server; это не INI-ключи. Для возврата к CPU features:

```bash
export LLAMA_DSPARK_GPU_FEATURES=0
```

Область применения: DeepSeek41 с DSpark, три последних trained taps, single sequence, causal/offload-KQV, CUDA layer split и CUDA encoder FC, batch<=8192 и ubatch<=512. При неподходящей конфигурации или отказе выделения памяти сохраняется host fallback. Смотрите `GPU feature bridge = enabled` и `producer-local features enabled`; `GPU feature packing fallback to host` сообщает возврат к CPU packing.

Raw owners определяются scheduler probe, а не номерами слоёв/GPU. Если producer меняется для короткого batch, выполняется peer fallback в запланированную plane с записью причины в лог. Host snapshot сохраняется. В target prefill не добавляется раннее ожидание encoder GPU; синхронизация target перед сборкой packet остаётся. Форматы GGUF/Engram и F32 семантика прежние.

При hidden5120/B8192/U512 retained buffers требуют 540.06MiB VRAM суммарно. В проверенном layout: CUDA4=320.04MiB,CUDA5=160.02MiB,CUDA0=60MiB; no_alloc/fit учитывают это распределение. Central bridge при LOCAL=0 требует510.02MiB на output GPU. CUDA peer transfers могут использовать внутренний host staging.

Проверки:33 target-native cases на frozen106K с exact taps/packed/selected logits/state/rollback и18 full-server запросов с одинаковыми token IDs/draft counters. На8K с Q3/MXFP8 Engram/Q8 KV/BF16 DSpark K3/full/p-min0.5,ctx100352 producer-local дал PP787.41 против772.35t/s у CPU reference (+1.95%), TG48.60/48.68 практически прежний. Один порядок CENTRAL->LOCAL->HOST; это bounded screen. Default включён по запросу пользователя. 2026-11-03 joint draft KV/logits/rollback/NLL и реальная память ctx176128 проверены отдельно (`1017-feature-local-joint-20261103`, `1017-feature-local-long-20261103`). LOCAL+ASYNC остаётся отдельной проверкой. Отчёт: `1017-feature-local-20261102` в validation/moe-cache проекта.

### Экспериментальный async ingress

ASYNC остаётся default OFF. Для воспроизведения прежнего central sync/async сравнения:

```bash
export LLAMA_DSPARK_GPU_FEATURES=1
export LLAMA_DSPARK_GPU_FEATURES_LOCAL=0
export LLAMA_DSPARK_GPU_FEATURES_ASYNC=1
```

Source owned stream упорядочивает copy до следующего pack, encoder ждёт copy-event; дополнительный payload сверх bridge не нужен. Непрямой/неподдерживаемый FC и несовпадение backend используют sync fallback. Token-input путь прежний; feature-update injection при generation тоже может использовать async.

Прежний central async screen сохранил token IDs/acceptance, но относительно central sync дал PP8K-0.43%,TG-1.88%. Ускорение не установлено; LOCAL+ASYNC совместно не проверены. Исторические central input-placement/async результаты сохранены в отчётах `1017-gpu-feature-transfer-20261102` и `1017-feature-ingress-async-20261102`.

### Prefill-skip M2

По умолчанию ON в поддержанных режимах, включая проверенный BF16 DSpark. Переменная окружения не требуется. Для отключения перед запуском сервера:

```bash
export LLAMA_DSV41_PREFILL_SKIP=0
```

Проверен target-only Q3/MXFP8 Engram/Q8 KV, P1, B8192/U512, TopK2048 и фиксированное полное GPU-размещение. Ранние целые chunks публикуют KV/index источника20 без ненужного верхнего query/FFN; сохраняется минимум3072 строк полного хвоста. Неполный последний chunk остаётся исходной формы. На server8K измерено PP812.51->1055.92t/s (+29.96%), TG37.77/37.81 практически одинаковый, token IDs совпали. При512 оптимизация не активируется.

Дополнительный target-only98K gate: полный prefix на тексте и C++ в unset/1/0 дал exact logits/seq/global/partial state, full/partial restore и rollback; NLL128 продолжения совпал отдельно на обоих корпусах. Проверены candidate-mask regime и две controlled compute-failure recovery (context clear + full-prefix retry). Fresh server warm+3 на98K: PP789.85->1210.43t/s (+53.25%), TG37.48->37.46, output IDs/model buffers exact, actual minfree1867MiB. Это один порядок OFF->ON на тексте, без sidecar; не общий benchmark качества или сравнение с Maya.

Теперь проверен и BF16 DSpark: передаются только корректные последние1024 строки features (для logical8192), старые позиции продвигаются существующим KV API без вычисления фиктивных признаков. Target сохраняет3584 строки полного хвоста, все реальные projections используют прежние512/partial shapes. Full-feature getter не выдаёт неполный массив как полный; ошибка seed очищает оба KV и требует повторить весь prefix.

Для проверенной конфигурации с sidecar используйте в preset:

```ini
batch-size = 8192
ubatch-size = 512
parallel = 1
cache-type-k = q8_0
cache-type-v = q8_0
spec-type = draft-dspark
spec-draft-n-max = 3
spec-draft-p-min = 0.5
spec-verify-policy = full
gpu-layers-draft = all
```

Проверенный target использует Q8_0 KV, Engram MXFP8, все routed experts на GPU и ctx100352. Размер контекста и размещение подбираются по VRAM; минимальный реальный запас в серверном тесте1445MiB. Режим требует достаточно большого logical batch; при batch-size2048 пропуска нет. LOCAL bridge включён штатно, ASYNC остаётся OFF.

С BF16K3/full/p-min0.5 warm+3: PP8K788.38->1045.25t/s (+32.58%), PP98K741.61->1065.66 (+43.70%). TG48.57->48.48 и46.22->45.92 соответственно; малые deltas из одного OFF->ON порядка не интерпретируются как установленное изменение TG. Все16 token arrays и draft counters совпали, model/compute budgets прежние, exit0. Native text/code8K/98K, full/partial restore, rollback и controlled seed-failure retry прошли; target/draft logits/state и NLL128 продолжения exact. CPU и central feature transport также проверены; speed относится к default LOCAL.

Поддержан один зарегистрированный DSpark consumer. Другие layer taps, embeddings/media, multiple sequences, eval/abort callbacks, LoRA/cvec, host experts/cache, async feature input и неподходящая topology сохраняют прежний путь. Actual OOM/exception/cancellation и другие model/KV/backend варианты этим набором не закрыты. Default ON; выключение: `LLAMA_DSV41_PREFILL_SKIP=0`. Unset или `1` включает автоматический выбор проверенного пути; другие явно заданные значения отключают его. Отчёты: `1017-prefill-skip-native-20261103`, `1017-prefill-skip-long-20261103`, `1017-dspark-tail-screen-20261103` и `1017-dspark-prefill-skip-20261103` в validation/moe-cache проекта.

## База знаний по оптимизации runtime

[HowTo, каталог механизмов, уроки и паспорт DS41](docs/runtime-optimization/README.md) хранятся в этой ветке вместе с выбранными исходными отчётами и воспроизводимыми fixtures. Для работы агента доступен [skill runtime-optimize](skills/runtime-optimize/SKILL.md).


## HC root/reshape: общая CUDA-копия, default ON

`GGML_SCHED_COPY_RESHAPE` теперь включён при unset или1. В подходящем CUDA pipeline соседний backend получает одну копию HC-активаций вместо двух для root и полного reshape. Guards сохраняют прежний путь при неподходящей геометрии, backend, consumers или callback. Дополнительных настроек preset для включения не нужно.

На проверенной Q3/MXFP8 Engram/Q8KV/BF16 DSpark K3/full/p-min0.5,ctx100352/B8192/U512/P1 PP512/8K/98K вырос на4.06/4.26/3.77% к среднему двух OFF-контролей. Combined GPU reserve уменьшился на640MiB. Target/draft logits/state/taps/rollback exact, deltaNLL0 отдельно text/code; заметного TG выигрыша не установлено. Это не обещание такого gain для других моделей/квантов/KV/media/parallel.

Для отключения перед запуском родительского router/server:

```bash
GGML_SCHED_COPY_RESHAPE=0 ./llama-server ...
```

Это environment variable, не INI/CMake option. [Отчёт и проверки](docs/runtime-optimization/experiments/1017-hc-copy-sharing-20261103/REPORT.md).


### HC sharing: проверка большого контекста

После включения defaultON выполнен отдельный actual ctx176128/172032+16 и повтор8K на фиксированной Q3/Q8KV/BF16K3 конфигурации B8192/U512/P1/TopK2048: все15360 routed bundles GPU,minimum761.25MiB свободно,exit0,continuation IDs exact. Это проверенный layout с overrides из отчёта, не новый универсальный preset/auto-fit262K. Новых пользовательских опций не добавлено.

[Отчёт](docs/runtime-optimization/experiments/1017-hc-long-indexer-20261103/REPORT.md).


## Эксперимент: локальный candidate pooling, пока OFF

`LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1` переносит CONT/padding/pool к GPU, где получены indexer scores. Последующий block selector остаётся на прежней карте. Блоки8/top2048blocks/final2048positions и вычисления не меняются. Задавайте environment до загрузки нового контекста; unset/0 сохраняет прежний путь. Это environment option, не строка INI.

```bash
LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 ./llama-server ...
```

Проверенный Q3/Q8KV/BF16K3 fixture: PP98K+1.46%,PP8K-0.28%; выигрыша TG нет,98K точка-1.07% требует повторов для оценки регрессии. Расчётный combined compute reserve меньше на~1073MiB при176K; actual176128/172032+16 прошёл с min761.25MiB и exact IDs. Флаг остаётся experimentalOFF, остальные неподтверждённые модели/режимы не объявляются проверенными.

[Отчёт](docs/runtime-optimization/experiments/1017-candidate-pool-local-20261103/REPORT.md).


## Компактная передача candidate mask: default ON

`LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1` передаёт существующие значения0/-inf по блокам и разворачивает прежнюю плотную маску локально на GPU-потребителях. Флаг фиксируется при создании контекста;unset/1разрешает путь при соблюдении guards,0отключает. Это переменная окружения перед запуском родительского router/server, не строка INI. События, protected copy slots и численная семантика сохранены.

Измеренная комбинация (предыдущий pooling тоже opt-in/defaultOFF):

```bash
LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=1 ./llama-server ...
```

На frozen Q3/MXFP8Engram/Q8KV/BF16DSparkK3/full/p_min0.5/B8192/U512/P1/TopK2048 PP98K1121.83->1154.72t/s (+2.93% к тому же runtime сpool1/block0). TG gain не установлен (-0.62%,singleton),8Kcandidateнеактивен.24rawtarget/draftgroups/two98Kcorpora exact,NLLdelta0;native27outputs/draftcounters exact. Actual176128/172032+16/repeat8K min1143.25MiB,все15360expertsGPU/placementexact,exit0. Compute reserve -520.52MiB при176K;wide candidate-mask API payload26.700GB->3.338GB,это не8xengine gain.

Проверенное размещение/guards и полные argv/env находятся в [отчёте](docs/runtime-optimization/experiments/1017-candidate-block-local-20261103/REPORT.md). Pool0 отдельно проверен только metadata, а не native speed; другие модели/KV/media/parallel>1/auto-fit262K требуют своих gates. BLOCK_LOCAL теперь ON при unset/1; предыдущий CANDIDATE_POOL_LOCAL остаётся OFF по умолчанию.

Для отключения до запуска родительского server/router:

```bash
LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL=0 ./llama-server ...
```

Promotion проверен fresh no_alloc default==explicit1 на100K/176K приpool0 и на100K приpool1;0сохраняет fallback. Нового native benchmark после однострочного изменения default не запускали; измеренная выше скорость относится кpool1/block1.


## Текущие candidate defaults: native gate закрыт

[Отчёт](docs/runtime-optimization/experiments/1017-pool0-default-20261103/REPORT.md): runtime `my_maya 0e6f3cec3` и defaults не менялись. Pooling не задан (OFF); в default arm BLOCK_LOCAL также не задан (ON). На двух 98K text/code корпусах 24 группы logits/taps/state/rollback совпали побитово, delta NLL=0; четыре seed failures/retry прошли. Native OFF/default/OFF: 27 output arrays и draft counters совпали. PP98K 1107.75 -> 1138.16 t/s (+2.75%); TG 46.21 -> 45.82 (-0.84%, по одному измерению на arm, ускорение TG не установлено).

Actual ctx=163840 / prefix=159744 / repeat 8K прошёл: минимум 679.25 MiB свободной VRAM, все 15360 экспертных bundles на GPU, placement совпал, exit 0. Context=176128 не допущен расчётом при pooling OFF; прежний actual 176128 с pooling=1 остаётся отдельной серией. Это fixed Q3/MXFP8 Engram/Q8 KV/BF16 DSpark K3/P1 конфигурация, не универсальный предел контекста.

Ошибка post-test сборки сводки (`queue.py` перекрыл stdlib) вызвала повтор metadata. Исходные precision/native/payload manifests сохранены побайтно; память и routes metadata сверены с исходными admission inputs. Incomplete manifest, старые hashes и recovery diagnostics сохранены в отчёте. Runner переименован и защищён от выполнения при импорте.

Далее: matched Maya comparison, свежий residual profile, затем Q2/Q4 MMQ. Media, parallel>1 и auto-fit 262K остаются отдельными проверками. Нового commit/deploy нет.

## Перекрытие CPU lookup и GPU prefill

`GGML_SCHED_CPU_LOOKUP_EARLY=1` разрешает ранний CPU GET_ROWS при доказанной независимости от предыдущего GPU split. По запросу пользователя режим теперь ON по умолчанию: unset/1 разрешает, 0 отключает. Он требует существующего pipeline с защищёнными input/copy slots; неподходящие графы сохраняют прежние ожидания. Это переменная окружения до запуска родительского server/router, а не строка INI.

```bash
LLAMA_PIPELINE_PREFILL=1 ./llama-server ...
```

При GPU-only tensor overrides по-прежнему нужен `LLAMA_PIPELINE_GPU_OVERRIDES=1`; перенос CPU-экспертов/staging этим не разрешается. На новой машине Q3/Q8 KV/TopK2048/U512/P1: PP8K +6,56%, PP98K +5,34%, PP512 без установленного выигрыша; общий выигрыш TG не установлен. Logits, KV, taps, rollback и NLL совпали с OFF. [Границы и артефакты](docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/REPORT.md). Исходные замеры выполнены на приватной библиотеке; новый default отдельно прошёл 42 проверки на рабочем модуле планировщика. Для прежнего порядка задайте `GGML_SCHED_CPU_LOOKUP_EARLY=0` до запуска server/router.
