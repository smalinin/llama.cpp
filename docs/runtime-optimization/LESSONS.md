# Практические уроки

## Доказательство должно соответствовать утверждению

| Наблюдение | Правильный вывод | Чего оно ещё не подтверждает |
| --- | --- | --- |
| Быстрое operator ядро | Кандидат заслуживает native screen | PP/TG gain с dispatch, copies и sync |
| Capture/replay действительно сработал | Branch и lifetime проверены в этом fixture | Существенное снижение wall time |
| Exact greedy IDs | Этот sampled trajectory совпал | Равенство logits и распределений |
| Один repeated greedy отличается | Есть численная/исполнительная вариация для расследования | Автоматически ошибка архитектуры или снижение качества |
| Seq snapshot совпал | Live sequence представление совпало | Full/global serialization старых SWA cells |
| no_alloc вписался | План считает такой budget | Runtime peak и отсутствие late growth |
| Acceptance выросла | Драфты чаще принимаются по этому знаменателю | Draft+verification стали дешевле на output token |
| Maya быстрее в старой серии | Есть ориентир для исследования | Причина разрыва или новый matched A/B |

## Ошибки, на которые уже потратили время

1. **Старый профиль после большого изменения.** M2 удаляет значительную часть upper FFN работы; прежнее распределение bottlenecks нужно измерить снова.
2. **Скрытый CPU staging.** GPU input tensor мог пройти через CUDA_Host. Нужны actual source/destination/stream и transfer evidence, а не только тип исходного buffer.
3. **Снятый fence без владения.** Input slots, packed features и retained taps должны переживать consumer reads; producer overwrite допускается после соответствующего event. Graph identity должна учитывать shapes/addresses/properties и fusion consumers.
4. **Неполная dependency cone.** KEEP128 недостаточен для M2. Seq-state мог совпасть при отличающихся masked physical cells; CPU proof включил отрицательный control и full/global snapshots.
5. **Недоступные признаки выглядели доступными.** После restore или failed decode capacity старого GPU buffer сохранялась. Getter должен проверять опубликованный valid range, а не одну allocation capacity; новый успешный decode заново публикует диапазон.
6. **Загружен другой sidecar/alias.** Проверять startup path и model/library hashes. BF16 и MXFP4 draft, разные K/p-min и template prefixes давали несопоставимые числа.
7. **Microbenchmark без полной стоимости.** FP16/GEMM с offline partition замедлился ещё до учёта online classifier. Новая идея должна учитывать counts/gather/conversion/scatter/launches и fallback.
8. **Суммарная VRAM скрывает локальный дефицит.** Между CUDA ordinal и NVML index нужен PCI/UUID mapping; spare slot, cache entry и whole expert bundle - разные единицы.
9. **Профилирующий callback выключил нужный путь.** В M2/pipeline есть callback guards. Диагностический run должен подтвердить eligibility до интерпретации таймингов.
10. **Успешный isolated screen расширили слишком далеко.** Ada/Ampere, один тип весов, P1 и text-only не подтверждают другую GPU, Q2/Q4, media или parallel>1. F16/Q8 memory tradeoff и same-Q8 kernel gain измеряются отдельно.

Отрицательное решение сохранять с узким scope: serial FP16/GEMM не прошёл, но grouped GEMM не проверен; FFN-only graph regions не ускорили PP, но это не отказ от всех CUDA graphs. Артефакты initial failures полезны для отладки стенда; их исправление не означает исправление runtime, если ошибка была только в harness.

Источники и исходные отчёты перечислены в [локальном архиве](evidence/README.md).

## Уточнения residual profile, 2026-11-03

- [Новый опыт](experiments/1017-m2-residual-profile-20261103/REPORT.md) показал, почему нельзя приписывать12s feature-getter самому draft: packed getter ждёт target, actual seed kernels занимают19ms. Вложенный packed getter должен вычитаться при вычислении exclusive bridge.
- Большой cudaEventSynchronize inclusive sum не равен ожиданию input slots. Узкая гипотеза alloc copy wait проверена отдельными counters и не подтвердилась; timer difference имеет собственные границы и overhead.
- Logical scheduler async copies могут породить host-staged CUPTI records. Два tensor names с одинаковыми bytes/count - повод проверить alias/actual transfers, а не готовое доказательство лишнего DMA.
- HTTP prompt8192 прошёл logical7676+512+4 и17 ubatches, поэтому seed rows1536 отличаются от direct8192 fixture1024. Рецепт dependency cone проверяется на реальных scheduler shapes.
- Cached build-info label может отставать от фактически собранного source. Runtime/library hashes, source identity и active branch evidence важнее одной строки версии.

## Уточнения actual memory и alias audit, 2026-11-03

- [M2 ctx176128](experiments/1017-m2-large-context-20261103/REPORT.md) подтверждён свежим payload172032+16, а не историческим LOCAL/no_alloc. Расчёт учитывает per-device measured overhead+64MiB; actual sampled floor601.25MiB. M2 сохраняет полный fallback reserve, поэтому одинаковые ON/OFF budgets ожидаемы.
- Pointer-only API interception видит unrelated temporaries при повторном использовании allocation address. Для root/reshape duplication нужны полный byte range, root/data/offset/stride equivalence, source/destination devices/stream и реальные destination consumers. В этом опыте32 full-range pairs подтверждены,12 partial groups исключены.
- Повторный CUDA API request ещё не равен измеренному physical DMA или PP gain. Объединение destination buffers меняет lifetime/allocator/fusion условия и нуждается в отдельном дизайне и exact/native gates.
- Post-request ошибка historical JSON schema может дать driver exit1 при server exit0. Не переписывать raw complete/error; сохранить initial script/trace и отдельно валидировать responses/memory/cleanup после исправления постобработки.

## HC sharing: что переносить в другой runtime

[Опыт](experiments/1017-hc-copy-sharing-20261103/REPORT.md): полная root/view byte equivalence недостаточна без same-split ownership и protected canonical copy. Новую destination view нужно включить в allocation graph перед consumer: gallocr считает n_views по nodes. В global copy cache alias не добавлялся, поэтому scope не расширяется на следующие split. Unknown/mutating readers исключены заранее.

- Сокращение requests2->1 подтверждено owned-backend counter, а native PP+4% проверен отдельно от profiler. Не превращать сумму CUPTI copy durations в прогноз gain.
- Combined target+bound-draft/taps reserve важнее standalone component: target-only выглядит как800MiB, после binding реально640MiB. Sampled global minimum может почти не вырасти, если его ограничивает другая GPU.
- ExplicitON correctness/native evidence и default-selector promotion - разные записи. Исходные measured patch/hashes сохранены; для unset==1 повторены operator/capture и no_alloc, без бессмысленного повтора всего NLL/98K набора.


## Indexer dataflow: отделять block selector от final TopK

[Свежий audit](experiments/1017-hc-long-indexer-20261103/REPORT.md) показал: source20 compression_ratio1, candidate2048blocks x8rows, final2048positions. Carry final source TopK вместо block mask меняет алгоритм. На другом checkpoint source mapping и ratios проверять из actual hparams, не переносить DS4 constants.

- Pool поддерживается CUDA, но без assignment hint CONT и pool оказались на следующей карте. Pinning только pool может сохранить полный score transfer через прежний CONT. Сначала рассматривать весь pre-pool chain, не удалять copy/node без lifetime gate.
- Scheduler может вызвать backend iface напрямую: пустой generic copy-wrapper счётчик не доказывает отсутствие copies. В этом опыте metadata-constrained next-consumer/size/API correlation покрыла72+72 expected requests; source/destination CUDA pointers не перехвачены.
- CUPTI activity record bytes и duration не равны аппаратным PCIe counters/critical-path стоимости. Nsight graph-ID и incomplete-event warnings нужно сохранить; selected matched records не доказывают полноту всего GPU trace. Два staged records полного logical size не превращать в измеренную bus bandwidth.
- DefaultON actual176128 с новым combined reserve измерен отдельно: min761.25MiB. Старый M2/copyOFF601.25MiB остаётся historical gate, а не доказательством текущего selector.
