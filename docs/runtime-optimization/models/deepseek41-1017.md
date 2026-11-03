# Паспорт: DeepSeek-V4.1-Flash-1017 в my_maya

Дата снимка: 2026-11-03. Runtime `6d038260e`. Это паспорт проверенной Q3 конфигурации, а не спецификация всех квантов/размещений DeepSeek. Границы взяты из current source и [E01](../evidence/README.md#e01).

## Архитектура и данные

| Свойство | Проверенный вариант |
| --- | --- |
| Target | deepseek41, 40 blocks, hidden5120, 384 routed experts/block, 6 picks, expert width2304 |
| Веса | Q3_K-AntirezLike mixed recipe, 10 shards; реальные types проверять по tensors |
| Engram | Две таблицы MXFP8 в shards8/9; lazy CPU/mmap, не полный upload таблиц в VRAM |
| Top-K indexer | Явный override2048 |
| KV в последнем M2 speed test | Q8_0 K/V; исходный CLI default F16 этим не менялся |
| KV topology M2 | Source20, upper suffix20..39 с shared persistent KV/index; W128 |
| DSpark | BF16, arch dflash, trained semantics DS41, taps37/38/39, Markov/confidence heads |
| Draft seed ring | Effective SWA129, physical capacity768; 3 stages, encoder input3x5120 |
| Execution | CUDA layer split, P1/text, B8192/U512, no media/callback/adapters/cache/staging |
| GPUs | 6, mixed Ada/Ampere; FASTEST_FIRST; logicalCUDA5 RTX3090 24GiB, остальные48GiB в этом layout |
| Target/draft placement | Frozen all15360 routed bundles GPU; draftCUDA5/CUDA0, GPU overrides из [manifest](../evidence/reproduction/dspark-prefill/bench-results.json) |
| Context speed fixture | 100352; reserve floor512MiB, measured server minfree1445.25MiB |

## M2: что именно сохраняется

Источник20 публикует всю необходимую KV/index history. Для target-only полного состояния получается bound3054, округлённый full tail3072 при U512. С DSpark используется другой cut: начало доступных features `floor((N-768)/512)*512`, и `cut <= first_valid - 18*127`, округление вниз512; также действует прежний target raw/output bound.

| Logical N | Cut с DSpark | Полный target tail | First valid tap row | Реальный injection tail |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 4608 | 3584 | 7168 | 1024 |
| 7676 | 4096 | 3580 | 6656 | 1020 |
| 7681 | 4096 | 3585 | 6656 | 1025 |

Это относительные rows текущего batch; абсолютные позиции/seq0 проверяются в seed. Нельзя заменить physical768 на live129 или всегда брать последние два ubatch. До корректного хвоста draft продвигает только metadata в закрытой транзакции, затем обычные512/partial projections перезаписывают всё кольцо. Ошибка seed очищает оба контекста и требует full prefix reload.

`LLAMA_DSV41_PREFILL_SKIP` default ON после решения пользователя: unset/1 разрешает guarded path,0 отключает. При B2048 skip не активируется. Иные consumers, topology, host experts/cache, media, parallel, eval/abort callbacks, LoRA/cvec и async feature input сохраняют fallback.

## Проверки и числа

CPU108 ring fixtures, short native24, long98K30 групп/36 rollback workflows/2 controlled failures, transport24 и final lifetime12. В пределах fixture target/draft logits/seq/global states, real taps,K3/K5/confidence, restore и replay exact. NLL128 продолжения отдельно text/code delta0; это не PPL всех prompt tokens. Последний final packed-range guard проверен отдельной сборкой после timed A/B; default promotion позднее меняет selector при unset.

С BF16K3/full/p-min0.5/backend sampling ON, default LOCAL, ASYNC OFF, warm+3 x256 greedy:

| Prefix | OFF PP | ON PP | OFF TG | ON TG |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 788.38 | 1045.25 | 48.57 | 48.48 |
| 98304 | 741.61 | 1065.66 | 46.22 | 45.92 |

Один OFF->ON порядок и один speed corpus. Small TG deltas не интерпретируются как доказанная регрессия/ускорение. Не сравнивать напрямую со старым Maya903.18/52.91: context/KV/placement/K/output limits и timer boundaries различались. Свежий matched Maya test остаётся открытым.

Полный reproduction bundle и неизменяемые measured/final patches находятся в [evidence](../evidence/README.md#e01). Старые helper scripts требуют remapping recorded paths и сборки с соответствующими internal headers. Прежний pre-M2 actualctx176128 дополнен свежим [M2176128 gate](../experiments/1017-m2-large-context-20261103/REPORT.md):172032+16, minfree601.25MiB, placement exact, exit0. Это fixed-layout memory/lifecycle проверка; свободный auto-fit262K не закрыт.

## Следующий профиль после M2

[Residual profile](../experiments/1017-m2-residual-profile-20261103/REPORT.md): native8K/98K controls1050.63/1069.16PP, по одному измеренному control, не новая A/B серия. NSYS сохранил IDs/draft counters и M2. Routed Q3/Q4: 54-57% kernel sum; CPU token/Engram2.027s на98K; seed kernels19ms. Следующая гипотеза связана с cross-GPU tensor traffic и возможными дублирующими view copies; whole-context176128 проверен последующим этапом, ссылка выше.

## Аудит HC view transfers

[Actual memory/alias report](../experiments/1017-m2-large-context-20261103/REPORT.md): на одном8K+16 request16 wide frames,32 root/reshape alias pairs. CUDA0->1 l_last-6 и CUDA1->2 l_last-14 получают по две peer-copy в разные destination buffers с общим src/device/bytes/stream. Всего1.342GB повторных API bytes; matched destination consumers проверены. Это не CUPTI physical DMA и не измеренный PP gain. Runtime пока без dedup fix; необходим design ownership/events/allocator/graph-fusion guards и отдельный correctness/native A/B.
