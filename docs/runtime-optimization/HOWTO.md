# HowTo: перенести опыт оптимизации на другую модель

## 1. Установить исходную конфигурацию

Заполнить [паспорт](templates/MODEL.md). Читать реальные GGUF metadata/tensor types и model graph: название семейства и размер файла не определяют вычислительный путь. Зафиксировать target, mmproj и draft отдельно. Снять commit, dirty patch, build options, хеши бинарника и загружаемых библиотек; helper с internal headers пересобирать для этого runtime.

Сохранить явный argv и environment после разворачивания preset, seed, sampling, template и готовые prompt token IDs. Для мультимодальности дополнительно фиксировать media hashes и preprocessing. Проверять из startup/request logs, какой файл и alias реально загрузились. Старые имена файлов и открытый preset могут не совпадать с процессом.

Измерить cold load отдельно от warm prefill/TG. Для рабочего baseline взять короткий prompt, длинный prompt, текст и код; длины выбрать по модели и задаче. В примере DS41 использованы 512/8192/98304, B8192/U512/P1. Это исторический набор, а не обязательные параметры новой модели.

## 2. Найти фактическое ограничение

Сначала сравнить no_alloc/fit с реальным placement и peak memory. Разложить VRAM на веса, резидентные expert projections/cache, KV/states/replicas, retained taps, compute workspace и driver overhead. Учитывать каждую GPU, а не только сумму. UUID/PCI bus ID связывает CUDA ordinal с NVML; разные порядки устройств могут скрыть перегруженную карту.

Для этой машины принят минимум 512 MiB actual free на каждой GPU; no_alloc прогноз сам по себе его не доказывает. Sampled NVML minima не гарантируют отсутствие пика между samples. Для другой машины бюджет согласовать до запуска и сохранить watcher/stop condition.

Профилировать отдельно prefill, decode и target verification: MoE/dense matmul, attention/indexer, lookup, H2D/D2H/peer copies, launches, allocations и scheduler waits. Считать critical path и overlap по timeline. Сумма длительностей параллельных kernels не равна wall time.

Убедиться, что профилирование не изменило eligibility. В DS41 eval/abort callbacks могут отключать M2/pipeline: похожие строки в профиле тогда описывают baseline. Отмечать активный branch и fallback reason; большие node counts и высокий graph reuse сами по себе не доказывают gain. После существенного изменения повторно снять остаточный профиль.

## 3. Выбрать один ограниченный механизм

Проверить [каталог](CATALOG.md) и [отрицательные результаты](LESSONS.md). Для гипотезы записать trigger, стоимость, зависимости, поддержанные формы и fallback. Сохранять batch, precision, KV, placement, Top-K и sampling между arms, если именно они не являются переменной опыта.

Перед переносом pruning вывести dependency cone из реальной topology: владельцы KV/index keys, окна каждого слоя, taps потребителей, physical ring capacity, sinks и требуемые outputs. Shared KV не означает, что можно пропустить весь верхний suffix для всех старых токенов. Гарантии live window и full serialized state различаются.

Для copies/reuse определить producer, consumer, buffers, streams, completion events и момент допустимой перезаписи. Полное удаление synchronize, UID-only reuse или игнорирование адресов не являются готовой оптимизацией. Подсчёт route histogram и классификация hot experts должны входить в runtime cost; подготовленные offline counts являются screening fixture.

## 4. Проверить корректность на подходящем уровне

- Оператор: реальная геометрия/types/strides, независимый reference, tails, пустые/повторные/неотсортированные selections, aliasing, finite outputs и canaries. Инструментировать изменённый код при memcheck/sanitizer.
- Полная модель: teacher-forced logits на одном prefix, raw taps, confidence и draft logits; sequence и full/global snapshots проверять отдельно. Сначала сверить сам reference и семантику restore.
- Lifecycle: causal masks, append, prompt/slot reuse, checkpoint/full/partial restore, rollback после нулевого/частичного/полного принятия и следующий decode/draft. В speculative batches убирать rejected/noise KV до canonical replay.
- Отказы: используемый контракт должен либо сохранять согласованное состояние, либо явно очищать его и требовать reload. Controlled failure не считать actual OOM/cancel/exception coverage.

Для layout/transport/reuse с прежними операциями стремиться к bit-exact результату. Если меняются округление/редукции/quantization, отдельно оценить max error, top-1 flips и NLL/PPL на тексте и коде, затем task-level ответы. Для DS41 был согласован диагностический ориентир +0.01 nat/token; он не является универсальным разрешением или автоматическим merge gate. Совпадение greedy IDs не доказывает равенство распределений, а расхождение IDs само по себе не объясняет причину.

## 5. Измерить полный runtime

Запускать чистый native A/B без profiler, sanitizers, trace callbacks, фоновой квантизации и параллельной сборки. Если нужна новая библиотека, завершить её сборку до запуска процесса. Использовать один прогрев и минимум три measured повтора как начальный screen; для небольшого эффекта проверить обратный порядок/ABBA и дополнительную выборку.

Сохранить один и тот же массив input IDs и фиксированный output limit/stop policy, каждый measured prefix пересчитать полностью. Подтвердить отсутствие prompt cache reuse по счётчикам. Отдельно показать PP, TTFT/wall time, TG, длину ответа, per-device memory и startup time. Сопоставить границы таймера, в том числе учёт первого output token. Для stochastic генерации использовать дополнительные seeds/выборку.

Для DSpark фиксировать K, admission/p-min/cumulative policy, backend sampling и placement draft. Показывать accepted/generated и per-position kept/reached с точным знаменателем. Одинаковая acceptance не гарантирует ускорения: draft и verification имеют собственную стоимость. Изменение K или длины ответа не выдавать за ускорение ядра.

Operator gain, калибровка или capture count не заменяют native PP/TG. Новая реализация не должна получить promotion только потому, что baseline был случайно размещён хуже или использовал CPU fallback. Для честного сравнения KV-форматов отдельно показывать эффект памяти/residency.

## 6. Сохранить решение

Заполнить [отчёт](templates/EXPERIMENT.md): passed/failed/inconclusive/not run для каждого gate. Указать один из статусов: measured candidate, shipped opt-in, shipped default, rejected, open. Решение о default отделить от измеренной source/binary identity. Сохранить точный patch/commit, library hashes, raw timings, inputs, manifests и короткий анализ ограничений.

При отрицательном speed gate прекратить расширение этого кандидата; сохранить его границы и условия повторного рассмотрения. Положительный результат на Q3/Ada не даёт автоматического допуска Q2/Q4, другой GPU, prefill, parallel или мультимодальности.
