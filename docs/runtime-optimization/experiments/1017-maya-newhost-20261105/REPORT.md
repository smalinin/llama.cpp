# Наш runtime и Maya: новое оборудование, 2026-11-05

**Серия завершена: четыре свежих процесса, 52 полных uncached запроса, exit 0.** Материнская плата, CPU, RAM и PCIe-порядок GPU изменились. Все таблицы ниже относятся только к новой машине. Скорости, калибровки и допуски памяти прежних серий не использовались как baseline. Runtime source и defaults не менялись; commit/push не выполнялись.

## Новая платформа и binaries

Threadripper PRO 3945WX, 12 ядер/24 потока, 192 GB RAM, одна NUMA node. Оба движка получили 12 CPU threads и одинаковый PCI-порядок GPU через явные UUID: пять карт 48 GB, последняя RTX 3090 24 GB. [Hardware snapshot](artifacts/lscpu.txt), [RAM](artifacts/meminfo.txt), [GPU UUID/PCI](artifacts/gpu.txt), [topology](artifacts/topology.txt), board/BIOS files в artifacts. GPU labels прежней платы не переиспользуются как адреса устройств.

Исходники нашего runtime на старте: `my_build 91c88beb2`. При финальной проверке checkout уже `my_maya` на том же commit; source/library hashes за время измерений не изменились. Сервер/CUDA унаследованы из проверенной сборки; embedded build-info может показывать предыдущий parent `b3377774b`. Точные source и library hashes сохранены в [config](artifacts/config.json). Наш CPU backend пересобран с AVX2/FMA/F16C, GGML_NATIVE=OFF. CPU-объекты Maya ggml пересобраны с `-march=native` на новом CPU; ISA AVX2 подтверждена, AVX512 отсутствует. Использован отдельный Maya executable с прежними CUDA objects и новой CPU library. Пользовательский `build/maya-ds4` не изменился. CMake logs/caches, CPU flags, native macros, compiler version и private link argv сохранены. Это замороженные binaries данной серии, не заявление об одинаковых compiler flags движков.

## Протокол

Оба движка читают одни десять SSD GGUF Q3_K-AntirezLike, включая MXFP8 Engram. Maya pack заново построен прямо из этих GGUF; weights не копировались и не менялись. Одинаковый sidecar по пути `dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf`: фактически 7.219 GB payload MXFP4, 1.262 GB BF16 и 0.020 GB F32, а не однородный BF16. [Типы](artifacts/tensor-types.json), SHA256 sidecar и stats каждого shard в config.

Одинаковые входные token IDs technical-prose, длины 512/8192/98304, context=100352, TopK=2048, chunk/ubatch=512, 256 greedy output tokens, temperature=0/top_k=1/top_p=1/min_p=0/repeat_penalty=1/seed=1234, EOS игнорируется. Между запросами sentinel token 97. У всех запросов server cache_n=0 / Maya reused=0, prefix прочитан полностью. TG вычислен по **255 decode intervals** у обоих движков: первый токен относится к prefill.

Порядок: наш OFF, Maya OFF, Maya ON, наш ON. На 512/8K по два warmup и три measured; на 98K один warmup и два measured. В таблицах медианы measured, все warmups и ranges сохранены. В native timing нет profiler/interposer/eval callback, сборки или параллельной калибровки.

Sidecar ON: max=5, individual confidence threshold=0.5, cumulative admission=0, empty cooldown=1; у нас verify policy full, backend sampling согласно frozen args. Наш runtime: B8192/U512, Q8_0 K/V, фиксированные GPU split/overrides, expert cache=0 и prefill staging=0. Это прежняя геометрия размещения, заново проверенная actual загрузкой и памятью; старые measured memory estimates не использованы. Maya: собственные KV и layer/expert placement, TP1/threads ON/reserve1024/mmap-experts auto/vision OFF. Старые lane-times не копировались; CPU/PCIe-калибровка снята заново в каждом private pack. Скопирован только профиль модельных частот экспертов для hot-start, без hardware timings. Исходный Maya pack/profile/config не изменён, hashes проверены.

## Скорость новой серии

Delta означает Maya / наш runtime - 1. Единицы PP/TG: tokens/s.

| Sidecar | Prefix | Наш PP | Maya PP | Delta PP | Наш TG | Maya TG | Delta TG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 512 | 736.30 | 678.15 | -7.90% | 43.16 | 48.16 | +11.59% |
| off | 8192 | 1356.56 | 2295.77 | +69.23% | 40.89 | 39.86 | -2.51% |
| off | 98304 | 1557.66 | 2520.09 | +61.79% | 40.87 | 38.73 | -5.23% |
| on | 512 | 720.21 | 673.24 | -6.52% | 40.94 | 68.53 | +67.41% |
| on | 8192 | 1345.49 | 2283.80 | +69.74% | 44.25 | 72.11 | +62.97% |
| on | 98304 | 1381.63 | 2527.36 | +82.93% | 45.99 | 61.49 | +33.69% |

Это end-to-end сравнение движков с разными KV, arithmetic и placement. Оно не изолирует эффект одного kernel или expert cache. На разных autoregressive продолжениях TG также зависит от маршрутов экспертов и поведения drafter; одинаковые input IDs не делают output workload идентичным.

## CPU и acceptance

CPU seconds - Linux process user+system time всех threads за полный запрос, включая PP/TG и обработку protocol. Mean CPU cores = CPU seconds / wall seconds; это не доля CPU expert math и не decode-only профиль. Acceptance - сумма accepted / offered по measured запросам, не среднее процентов.

| Sidecar | Prefix | Наш CPU,s | Maya CPU,s | Наш cores | Maya cores | Наш acceptance | Maya acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 512 | 6.68 | 11.61 | 1.01 | 1.92 | - | - |
| off | 8192 | 12.41 | 28.05 | 1.01 | 2.83 | - | - |
| off | 98304 | 69.90 | 255.88 | 1.01 | 5.61 | - | - |
| on | 512 | 7.03 | 9.75 | 1.00 | 2.26 | 61.21% | 65.11% |
| on | 8192 | 11.99 | 24.99 | 1.00 | 3.51 | 66.53% | 79.63% |
| on | 98304 | 77.27 | 254.86 | 1.00 | 5.92 | 77.72% | 75.61% |

Больший расход CPU не доказывает лучшее исполнение CPU FFN. В него входят workers, dispatch и ожидания; их вклад эта серия не разделяет. Per-position kept/reached Maya сохранены в raw log и results. Acceptance на разных generated prefixes не является прямой приёмкой качества drafter bridge; для неё нужны paired logits/drafts на одинаковом prefix.

## Память и размещение

У нашего target GPU-only expert overrides и offload 41/41; routed experts размещены на GPU. CPU_Mapped buffers также имеются, включая Engram и mapped views плотного shard; размер mapped buffer не равен объёму CPU expert compute. Полные model buffer строки сохранены. У Maya без sidecar 92 target experts на CPU lane (15268 target experts в GPU pool), с sidecar 153 (15207 target experts в GPU pool). Drafters добавляют свои experts; нельзя считать общие group counts вместе с MTP как число target experts. Финальные per-GPU layout и STAT tiers сохранены в results/logs. Общий resident payload и размещение между движками не идентичны.

| Arm | Sampled minimum free VRAM, MiB |
| --- | ---: |
| ours-off | 1795.12 |
| maya-off | 871.12 |
| maya-on | 867.12 |
| ours-on | 1271.25 |

Все четыре процесса завершились с exit 0, без recorded CUDA errors, allocation mismatch/OOM, forced kill или нарушения 512 MiB floor. NVML sample 100 ms / watcher 200 ms не исключают пик между samples. Контекст 100352 не является тестом 262K и не задаёт универсальный memory limit. Старые no_alloc/actual-context результаты с другой машины сюда не перенесены.

## Повторы и границы

Число уникальных output sequences / число measured повторов и PP ranges:

| Sidecar | Prefix | Наш unique/runs | Maya unique/runs | Наш PP range | Maya PP range |
| --- | ---: | ---: | ---: | --- | --- |
| off | 512 | 1/3 | 2/3 | 729.71..748.44 | 677.52..678.51 |
| off | 8192 | 1/3 | 1/3 | 1353.26..1361.15 | 2294.81..2324.63 |
| off | 98304 | 1/2 | 2/2 | 1556.99..1558.32 | 2519.71..2520.47 |
| on | 512 | 1/3 | 3/3 | 714.04..729.98 | 673.15..673.33 |
| on | 8192 | 1/3 | 2/3 | 1340.81..1345.56 | 2167.60..2299.51 |
| on | 98304 | 1/2 | 2/2 | 1378.01..1385.25 | 2497.70..2557.01 |

Серия измеряет скорость, не NLL/PPL, logits/state/rollback correctness. Отличие повторных greedy output требует отдельной диагностики и не даёт объявить движок принятым по качеству. Причины различий, включая CPU/GPU rounding и placement/cache, по одним timings не определяются. Фиксированный K5/full/cumulative=0 не является подбором лучшего n_max или adaptive policy для каждого движка. Один arm order, малое число повторов, особенно два на 98K, и разные generated continuations ограничивают выводы о малых deltas. Исторические проценты не складываются и не сравниваются с этой серией.

## Дальнейшие шаги

Следующий этап - свежий профиль prefill 8K/98K и target verification на этой платформе: GPU overlap между chunk512, host waits/dispatch, routed MMQ/attention/Engram и упаковка KV. Любой новый кандидат должен пройти свежие native speed/memory и согласованные logits/state/rollback/NLL gates на новой машине. Этот отчёт не объявляет перенос Maya целиком или изменение defaults.

[Полные результаты](artifacts/results.json), [сводка](artifacts/summary.json), [очередь](artifacts/run.log), [runner](artifacts/run-comparison.py), [подготовка](artifacts/prepare.py), [SHA256](artifacts/SHA256.json). Compiled private Maya binary локально в artifacts/bin и исключён из Git; его hash и link argv сохранены. Все собственные GPU-процессы завершены.
