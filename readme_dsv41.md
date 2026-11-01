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
Q8 indexed backend уже поддерживает counts на NVIDIA Ampere/Ada: operator tests и CUDA memcheck прошли. Encoded Q8 row-copy API (CPU/CUDA) и связка packing -> attention проверены побитово. Подключение к native compact graph, специализированный MMA512, полный lifecycle и quality gate ещё в работе.
Экспериментальный `LLAMA_DSV41_Q8_PACK=1` сохраняет encoded Q8 при raw ordering
и короткой verification packing (2–6 tokens). По умолчанию OFF. Работает в
ограниченном single-sequence CUDA layer split; causal masks и learned sinks сохранены.
Same-Q8 logits/state/taps/rollback совпали на1200 позициях512/8K. Masked FA пока
распаковывает KV в F16 внутри CUDA; compact Q8/MMA512 и полный quality gate ещё не готовы.
Backend prototype `GGML_CUDA_DSV41_Q8_MMA=1` (default OFF) добавляет shared K=V
MMA512/GQA64 с sinks/counts. Native Q8 compact graph ещё не подключён. Общий
F16 tile64 подключён отдельно через `GGML_CUDA_DSV41_F16_PREFILL64=1` (default OFF):
только compact F16 K=V, полный chunk512, single stream, head512/GQA64, sinks/counts,
Ampere/Ada. Порог24576 compressed-записей сохраняется; короткий decode/verification
использует прежний путь. `GGML_CUDA_DSV41_F16_PREFILL64_TRACE=1` и `--verbosity4`
показывают выбор kernel. Limited native screen после98K истории дал tail PP+9,8–11,8%,
TG без явного изменения; replay/rollback PASS. ΔNLL heldout text−0,00945/code+0,00711
на257 labels/corpus; logits/taps не bit exact. Нужен более широкий quality/sidecar gate
до default ON. Q8 prototype по-прежнему требует native compact integration.
Короткий code NLL screen показал +0.033 nat/token против F16 при ориентире+0.01;
это не полный PPL-тест, и рекомендовать Q8 вместо F16 пока рано.
