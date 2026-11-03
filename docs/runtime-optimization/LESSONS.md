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
