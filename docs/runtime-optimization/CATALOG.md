# Каталог механизмов и границ применимости

Состояние на runtime `6d038260e`, 2026-11-03. Shipped/default означает наличие dispatch в этой ветке с guards. Нужные model types, размеры, backend и свободная память всё равно проверяются. Исторические проценты относятся только к перечисленным опытам.

| ID | Механизм и статус | Что можно переиспользовать | Что установить для новой модели | Локальное доказательство |
| --- | --- | --- | --- | --- |
| O1 | CPU MXFP8 LUT, shipped default | Табличный decode с сохранением scale/exception semantics | Реальный тип таблицы MXFP8, CPU GET_ROWS, exceptional blocks; Q5_K/Q8_0 этот LUT не ускоряет | [E02](evidence/README.md#e02) |
| O2 | Routed Q3_K MMQ tile32, shipped default в подходящем dispatch | Выбор тайла по фактической форме и GPU, whole-K reduction | Число экспертов, rows/expert, chunk, types, Ada/Ampere; Q2/Q4 и другие GPU требуют своих gates | [E03](evidence/README.md#e03) |
| O3 | GPU-only overrides + pipeline input ownership, shipped в ограниченном DS41 пути | Eligibility по actual buffers, несколько input copies, reuse после completion event, metadata/real parity | Host spill, cache/staging, consumers, split, offsets и scheduler lifetimes; разрешённый override не доказывает overlap | [Текущий код](../../src/llama-context.cpp), [режимы](../../readme_dsv41.md) |
| O4 | Encoded Q8 packing, compact/counts MMA, dual raw/compressed sources; shipped conditional | Бounded unpack/gather вместо обработки всей history, явные counts/masks, shared KV reads | Q8 layout и rounding, head geometry, sinks, Top-K, padding, masks, GPU и reserve; F16 reference и same-Q8 reference различать | [E04](evidence/README.md#e04), [E13](evidence/README.md#e13), [README](../../readme_dsv41.md) |
| O5 | Batched target verification и sample batching, shipped conditional | Сократить launches для нескольких independent queries, сохранить causal/reduction semantics | Draft block width, all-logits outputs, rollback capacity, layout, kernels; plain GEMM может менять rounding | [README](../../readme_dsv41.md), [source](../../common/speculative.cpp) |
| O6 | Producer-local DSpark taps, shipped default; ASYNC отдельно OFF | Хранить данные у реального producer, собирать нужный chunk у consumer | Tap owners после scheduler, encoder device, copy costs, peer fallback, events и metadata budget; новый consumer требует собственного контракта | [E05](evidence/README.md#e05) |
| O7 | M2 source-KV-only prefill + DSpark tail-seeding, shipped default | Dependency-cone pruning и явный suffix contract; продвигать metadata только внутри закрытого seed с полным overwrite кольца | Точная KV/index topology и все consumers; DS41 source20/W128/taps37-39/C768 не универсальны; U512/P1/B<=8192/allGPU guards | [E01](evidence/README.md#e01), [паспорт](models/deepseek41-1017.md) |
| O8 | Mixed expert cache, byte budget/profile/spare; cache включается отдельно | Проверка publication/lifetime, actual host projection bytes, popularity profile, bounded replacement | Geometry gate/up/down, host/GPU routing, cache owner, полезные/физические slots, prefix/decode route distributions | [README](../../readme_dsv41.md), [source](../../src/llama-moecache.cpp) |
| O9 | Startup CPU/staging calibration и auto selection, opt-in auto | Измерить конкретные lanes и переиспользовать cost cache с fingerprint | Actual CPU/PCIe/GPU/types/shapes/workspace, stale invalidation, overhead calibration; текущий host-Q3 staging scope не универсален | [E06](evidence/README.md#e06) |

## Результаты, которые стоит помнить

- O1: CPU decoder быстрее примерно в4 раза в своём screen, но whole-model PP8K вырос лишь на +0.49..0.82%; PP512 +4.24%. Доля операции ограничивает общий gain.
- O2: Q3/BF16/F16-KV native B8192/U512 PP512 +28.09%, PP8K +24.39%; TG практически прежний. Это tile0->32 на прежнем build, а не дополнительный gain к последнему M2.
- O6: LOCAL PP8K 787.41 против CPU772.35 и central761.65 в одной fixed-order серии. Поздний98K LOCAL741.51 против CPU730.44; это небольшой transport эффект, а не ускорение всего draft на проценты transfer microbenchmark.
- O7: на Q3/Q8KV/BF16 K3 PP8K 788.38->1045.25 (+32.58%), PP98K741.61->1065.66 (+43.70%). TG48.57->48.48/46.22->45.92; exact response IDs/draft counters. Подробнее [E01](evidence/README.md#e01).

## Кандидаты, не прошедшие promotion

| ID | Кандидат | Результат и следующий разумный шаг |
| --- | --- | --- |
| N1 | Последовательный per-expert FP16/GEMM | Aggregate operator throughput при threshold64 Ada -18.38%, Ampere -2.18%; offline classification даже не включена в cost. В runtime не перенесён. Новый grouped/GPU-driven вариант оценивать отдельно. [E07](evidence/README.md#e07) |
| N2 | FFN-only CUDA graph regions | Exact correctness и реальные replays; PP512 +0.16%, Wiki8K -0.02%, code8K +0.03%. Полезный PP gain не установлен; не интегрирован. [E08](evidence/README.md#e08) |
| N3 | Центральный GPU bridge вместо CPU-overlapped reference | Несколько screens дали около -1.5% PP. Transport route и fences оказались важнее packing kernels; LOCAL стал отдельным кандидатом. [E09](evidence/README.md#e09) |
| N4 | ASYNC ingress поверх central bridge | PP8K -0.43%, TG -1.88% в одном порядке; ASYNC остаётся OFF. LOCAL+ASYNC отдельно не закрыт. [E10](evidence/README.md#e10) |
| N5 | Engram lookup ownership/prologue как самостоятельный PP фикс | Correctness прошёл, PP8K +0.54% text/+0.06% code; заметный gain не установлен. Новый вариант должен показать полезное overlap, а не перенести ожидание. [E11](evidence/README.md#e11) |

Отдельное сравнение F16/Q8 сохранило memory advantage Q8, но TG был ниже при том же placement; [E14](evidence/README.md#e14). Same-Q8 kernel gain и F16->Q8 tradeoff - разные эксперименты.

## Открыто

Q2/Q4 MMQ, другой model graph, multimodal/parallel>1, actual OOM/cancellation recovery и свободный joint auto-fit262K не закрываются перечисленными DS41 gates. Historical evidence различает controlled compute failure и реальные ошибки памяти. Для другого контекста/pipeline placement повторить reserve/actual memory проверки.
