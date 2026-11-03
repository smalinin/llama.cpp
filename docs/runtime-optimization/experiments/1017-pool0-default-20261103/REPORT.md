# DS41: проверка текущих candidate defaults, 2026-11-03

Проверки завершены на `my_maya`, commit `0e6f3cec3`: `LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL` не задан (ON), `LLAMA_DSV41_CANDIDATE_POOL_LOCAL` не задан (OFF). Runtime, библиотеки, веса, presets и defaults не менялись. Предыдущие результаты с pooling=1 остаются отдельной серией.

## Условия

Q3_K-AntirezLike, MXFP8 Engram, target KV Q8_0, BF16 DSpark, K=3/full/p-min=0.5, backend sampling ON, TopK=2048, B=8192, U=512, P=1, 20 CPU threads. Для precision helper использовались K3/K5 и raw sampling OFF. Все 15360 экспертных bundles размещены на GPU; split и overrides фиксированы. Fit, expert cache, prefill staging и mmproj отключены. Контекст speed/precision: 100352. Под defaults здесь понимаются только два candidate-флага, а не все параметры запуска.

Полные argv, environment, IDs и source/library hashes сохранены в [config](artifacts/config.json) и manifests. Контроль задаёт только BLOCK_LOCAL=0; default arm не задаёт ни одного из двух флагов. Значение фиксируется при создании контекста, поэтому arms запускались в свежих процессах.

## Точность, state и rollback

Два корпуса text/code с prefix 98304: 24 группы сырых результатов совпали побитово между OFF/default и сохранённым эталоном. Проверены target/draft logits, taps 37/38/39, seq/global/partial state, K3/K5/confidence, append 1/7/511/513, restore/reuse, zero/partial/full rollback. Delta NLL на 128 labels на корпус и режим равна 0; это диагностическая выборка, не full-corpus PPL.

Четыре контролируемых seed failures/retry прошли: по два подтверждённых injection и log marker на процесс, последующее состояние совпало с эталоном. Оба процесса завершились с exit 0 без allocation mismatch/OOM/CUDA errors; sampled minimum свободной памяти 1325.25/1327.25 MiB, выше 512 MiB.

Trace подтверждает remote pool на CUDA3 при source на CUDA2 в обоих arms. OFF: local REPEAT 0, legacy dense 166; default: 166 активных графов, 332 local REPEAT, legacy dense 0. Producer-local pooling не включался.

В прежнем helper внутри corpus loop принудительно устанавливался pooling=1. Только новая копия helper исправлена для проверки реальных defaults. Fault injection теперь зависит от явного тестового запроса, а не от pool-mode. Исходные runtime-файлы не менялись. [Precision manifest](artifacts/precision-results.json).

## Native speed: OFF/default/OFF

Три свежих процесса, 27 полных uncached запросов, cache_n=0, по 256 greedy output tokens. На arm: 512 warmup + 2 измерения, 8K warmup + 3, 98K warmup + 1. В timed runs нет profiler, LD_PRELOAD, eval callback, calibration или сборки. OFF ниже - среднее медиан двух контрольных arms.

| Prefix | OFF PP, t/s | Default PP, t/s | Delta PP | OFF TG, t/s | Default TG, t/s | Delta TG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 571.31 | 568.43 | -0.50% | 43.37 | 43.32 | -0.12% |
| 8192 | 1082.99 | 1084.81 | +0.17% | 48.56 | 48.35 | -0.45% |
| 98304 | 1107.75 | 1138.16 | +2.75% | 46.21 | 45.82 | -0.84% |

Все 27 output arrays и draft generated/accepted counters совпали, в том числе с сохранёнными pool=1 controls. Это сравнение block OFF/default при pooling OFF; историческая скорость pool=1 не является matched baseline этой серии. Model buffer logs совпали. Все exit 0, sampled minimum 1435.25 MiB, без warnings/floor breach.

На 512/8K candidate-ветвь неактивна; небольшие изменения не следует считать её ускорением. На 98K по одному измерению на arm и один порядок arms: ускорение TG не установлено, небольшое снижение требует осторожной интерпретации. OFF PP drift на 98K составил +0.16%. Диапазоны и повторы: [native manifest](artifacts/bench-results.json). Проценты разных серий не складываются.

## Память и длинный запуск

Metadata проверены для 100352 OFF/default и 176128/163840/159744 default. Model/KV budgets и per-tensor routes совпали; pool CUDA3, source score CUDA2, local repeats CUDA3/4. Для admission использован max observed overhead по каждой GPU из текущего 100K benchmark и исторического actual 176K с pooling=1, плюс guard 64 MiB. NVML/backend mapping сверено по UUID/PCI. Estimate не заменяет реальный запуск.

| Context | Допущен расчётом | Guarded minimum free, MiB |
| ---: | --- | ---: |
| 176128 | Нет | 405.25 |
| 163840 | Да | 614.59 |
| 159744 | Да | 684.38 |

Выбран ctx=163840 без изменения placement или включения pooling. Реальный запуск: 8192+256, 159744+16, повтор 8192+16. Все 15360 экспертных bundles на GPU, routed host bytes=0; placement совпал с metadata. Минимум свободной памяти **679.25 MiB**, exit 0, без OOM, allocation mismatch, CUDA errors, fallback или forced kill.

8K IDs совпали с benchmark; повтор совпал с первыми 16 IDs. Для prefix 159744 нет прежнего token oracle: это memory/lifecycle gate, а не отдельное доказательство long-prefix quality. Качество на 98K проверено выше. Loader-only placement observer не добавляет forward/CUDA API probes, GPU buffers или синхронизации. NVML 100 ms/watcher 200 ms дают sampled floor и не исключают пик между samples.

Прежний успешный actual 176128 с pooling=1 относится к другой конфигурации и не подтверждает память 176128 при pooling OFF. [Admission](artifacts/payload-admission.json), [actual results](artifacts/payload-163840/results.json).

## Ошибка стенда при сборке итогов

После завершения GPU-серий локальный `queue.py` перекрыл stdlib module при импорте и повторно запустил metadata-проверки. Повторная очередь остановлена; runner переименован в `run-gates.py` и защищён `if __name__ == '__main__'`. Новые inference benchmarks не запускались. Runtime и модели не менялись.

Precision, native и actual payload manifests сохранились побайтно: [integrity check](artifacts/summary-recovery/raw-results-integrity.json). Metadata runner manifest перезаписан и остался incomplete; он сохранён без исправления статуса. Некоторые metadata elapsed timings/hashes изменились, trace содержит повторные append records. Исходные bytes этих metadata не восстановлены.

[Отдельный semantic validator](artifacts/metadata-validation.json) проверил все пять complete metadata JSON: каждый per-device projected-free совпал с исходными входами, сохранёнными в admission, routes совпали между режимами и с actual placement. Из appended trace проверены только множества форм/владельцев, не execution counts. Исходные admission hashes сохранены, расхождения не скрыты. [Диагностика](artifacts/summary-recovery/stopped-processes.json), [validator source](artifacts/recover_metadata.py), [raw incomplete manifest](artifacts/metadata-runs.json).

## Границы и дальнейшая работа

Native gap для текущих defaults закрыт в указанной Q3/Q8 KV/BF16 K3/P1 конфигурации. Изменения runtime для прохождения проверок не потребовались. Другие модели, KV/media, parallel>1 и auto-fit 262K этим не проверены. Свежий matched Maya comparison, residual profile и Q2/Q4 MMQ остаются следующими отдельными этапами. Нового commit или deploy нет; собственные GPU-процессы завершены.

[Воспроизведение](artifacts/README.md), [provenance](artifacts/provenance.json), [SHA256](artifacts/SHA256.json), [сводка](artifacts/summary.json).
