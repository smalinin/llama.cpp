# Этап 17: свободная генерация с согласованными compressor и routed FFN

Этап 16 принят и закоммичен как `903560a8f`. Новый общий callback проверен в изолированном сервере. Все восемь ответов DSpark N=1 совпали с native; N=3 совпал в шести из восьми. Оба ответа explain при N=3 расходятся на индексе 244. Поэтому общая совместимость остаётся открытой, несмотря на точное совпадение baseline logits и прохождение трёх prompt при N=3.

Дополнительный target-only replay воспроизводит ту же развилку explain при width 3 без draft и rollback. Первая численная разница возникает раньше, при пересечении batch абсолютной позиции 256. Это определяет следующую узкую область для capture. Производственные исходники, установленный сервер и профили не менялись. Остановка для review перед коммитом этапа 17.

## Общая реализация сервера и replay

Изолированный `experiment-bin` копирует 32 файла неизменяемого snapshot этапа 8. Заменён только `libllama-server-impl.so`. Перекомпилирован временный `server-context.cpp`; сохранены flags, команды, SHA исходников и повторно использованных объектов/static libraries.

`diagnostic-callback.h` состоит из побайтно сохранённого candidate callback этапа 16 и cached scalar down helpers этапа 15, помещённых в тот же namespace. Этот единственный header используется сервером и replay. В нём согласуются floating precision, scalar FA, routed up/gate, HC/router, compressor KV/gate и routed down/weight/sum.

Callback действует только во время target generation, при width не более 4 и одном слоте. Draft context получает пустой callback. После каждого decode восстанавливается исходная precision узлов, очищаются cached down inputs; следующий prefill получает исходную арифметику. В журнале сохранён префикс `DS14_EVENT` для прежнего parser.

На baseline teacher-forced истории выполнены четыре replay widths 1/2/3/4. Все 95 полных векторов logits совпадают с native побайтно, включая впервые проверенную width 3. Общий SHA: `188c8690abade05294ac66643c56bc7727b21adb3ad6a65b5061e6ca0dfaed5b`.

Источники: [prepare/build](stage17/prepare-build.py), [source patch](stage17/server-experiment.patch), [общий header](stage17/diagnostic-callback.h), [build manifest](stage17/build-manifest.json), [replay manifest](stage17/replay-manifest.json), [SHA-контроли](stage17/replay-control-summary.json).

## Серверные запросы

Пять конфигураций запускаются последовательно отдельными процессами: исходный snapshot off, интеграция off без включённых замен, candidate off, candidate DSpark N=1 и N=3. Сохранены шесть GPU в прежнем порядке, split `1,1,1,1,1,0.4`, ctx 8192, batch/ubatch 2048/512, F16 target/draft KV, 12 потоков, CUDA Graphs disabled. Используется исходный MXFP4 sidecar; draft confidence threshold 0. Установленный сервер в тестах не участвует.

В каждой конфигурации четыре prompt: исходный baseline, SVG pelican/bicycle, Python merge intervals и explain database index. Для каждого два запроса: `n_probs=5` и `n_probs=0`, temperature 0, seed 1234, limit 256, `cache_prompt=false`. Это контроль вывода вероятностей, а не два идентичных timing-повтора. Во всех 40 запросах `cache_n=0`; сравниваются полные token IDs, текст и stop metadata до EOS или лимита.

Native baseline завершился по EOS на 95 токенах. Остальные три native-ответа достигли лимита 256. Все четыре native-ответа совпали с эталонами этапа 9. Все 16 сравнений integration-off/candidate-off с native точны. Все 20 пар `n_probs=5/0` совпали внутри своей конфигурации.

| Prompt | DSpark N=1, два запроса | DSpark N=3, два запроса | Первое отличие N=3 |
| --- | --- | --- | --- |
| Baseline | оба совпали | оба совпали | нет |
| SVG | оба совпали | оба совпали | нет до лимита 256 |
| Python | оба совпали | оба совпали | нет до лимита 256 |
| Explain | оба совпали | оба отличаются | индекс 244 |

На проверенных данных 14 из 16 speculative-ответов совпали с native, против 0 из 16 у сочетания этапа 14. Это улучшение проверенного поведения, не оценка качества ответов или скорости. В explain native выбирает ` conditions` (ID 4132), N=3 выбирает ` operations` (ID 7574), после общего текста `JOIN`. Native gap этих двух logits на индексе 244 около `+0,268639`.

Источники: [runner](stage17/run-free-generation.py), [off-контроли](stage17/free-control-summary.json), [полный анализ](stage17/free-summary.json), [native explain](stage17/free-runs/snapshot-off/explain-greedy-1-response.json), [N=3 explain](stage17/free-runs/candidate-n3/explain-greedy-1-response.json).

## Счётчики и acceptance

Все prefill-вызовы интегрированного сервера имеют нулевые precision/attention/upgate/matmul interventions. Scalar generation не заменяет attention/upgate/matmul. При generation width 2..4 ожидаются `40*width` attention, `40*width` routed up/gate и `167*width` matmul-результатов: 80 HC, 40 router, 40 routed down и 7 compressor projections. Все события соответствуют этим значениям и имеют `ret=0`.

| Конфигурация | Generation calls по width | Attention queries | Routed up/gate token-layers | Matmul token-results |
| --- | --- | ---: | ---: | ---: |
| Candidate N=1 | width 1: 2; width 2: 1214 | 97120 | 97120 | 405476 |
| Candidate N=3 | width 1: 4; width 2: 4; width 3: 4; width 4: 1036 | 166560 | 166560 | 695388 |

Acceptance ниже взят из запроса `n_probs=0`. Это доля принятых draft-токенов среди предложенных; её нельзя приравнивать к ускорению.

| Prompt | N=1 accepted/drafted | N=1 acceptance | N=3 accepted/drafted | N=3 acceptance |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 31/63 | 49,21% | 39/165 | 23,64% |
| SVG | 85/169 | 50,30% | 113/423 | 26,71% |
| Python | 75/180 | 41,67% | 98/465 | 21,08% |
| Explain | 60/195 | 30,77% | 85/507 | 16,77% |

Request timings сохранены в manifests/responses. Они включают исходные wide operations, повторные scalar computations, CPU-копии, allocation и synchronization. Этот callback не является производственным оптимизированным графом; его throughput не оценивает стоимость будущих CUDA kernels. Новый production benchmark не выполнялся: общая совместимость не пройдена.

## Target-only explain: первая численная разница до argmax

Для дополнительного контроля используется тот же native explain prompt из 23 токенов и все 256 native token IDs. Prefill точно повторяет `19/4`. После initial logits выполняются 255 teacher-forced входов; draft, rollback и dumps промежуточных тензоров отсутствуют. Callback не делает дополнительных capture asks. Выполнены восемь replay: narrow compressor и float2d, каждый при widths 1/2/3/4.

Оба scalar replay воспроизводят все 256 native argmax и имеют один SHA `08c5eeabd45ac1e4f4adeac6073b3f06a4028e524c200e28e19940c1acd3db19`. Широкий float2d-контроль даёт побайтно те же полные logits, что narrow, на каждой соответствующей ширине. Добавление обычных floating 2D пересчётов здесь не устраняет остаточную разницу.

| Width | Первое численное отличие, выходной индекс | Max / RMS всех logits | Другие native argmax | Gap conditions - operations на 244 |
| ---: | ---: | ---: | --- | ---: |
| 1 | нет | 0 / 0 | нет | +0,268639 |
| 2 | 233 | 1,627832 / 0,035900 | нет | +0,050053 |
| 3 | 232 | 3,081662 / 0,061278 | 244 | -0,130009 |
| 4 | 233 | 1,627832 / 0,035900 | нет | +0,050053 |

Width 3 воспроизводит тот же другой токен 7574 на индексе 244 без draft и rollback. Следовательно, для этой развилки они не являются необходимыми условиями. Это не исключает дополнительных эффектов speculative state. Widths 2/4 совпадают побайтно друг с другом; сохранение argmax на этих ширинах не означает сохранение logits.

Все предыдущие полные строки до указанного первого отличия совпадают. Выравнивание первых различающихся query:

| Width | Выходной индекс | Входной индекс | Начало batch | Колонка | Абсолютные позиции batch |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 232 | 231 | 231 | 0 | 254..256 |
| 2 | 233 | 232 | 232 | 0 | 255..256 |
| 4 | 233 | 232 | 232 | 0 | 255..258 |

Каждый такой batch впервые включает абсолютную позицию 256, в то время как его первая query ещё находится перед ней. Это указывает на границу K/V allocation/padding как проверяемую гипотезу. Сами K/V dimensions, маски и выходы attention вокруг этой границы в этапе 17 ещё не снимались; конкретная операция не установлена. Совпадение baseline95 не покрывало этот сценарий: baseline начинался с prompt длины 1655 и другой области cache.

Источники: [explain harness](stage17/explain-replay.cpp), [build manifest](stage17/explain-build-manifest.json), [run manifest](stage17/explain-replay-manifest.json), [анализ](stage17/analyze-explain.py), [результат](stage17/explain-summary.json), [native IDs](stage17/explain-native.i32).

## Решение и следующий шаг

Завершены пять серверных процессов, 40 completion requests, 15 template requests и 12 модельных replay. Все процессы завершились с кодом 0. Четыре baseline95 SHA-контроля точны; оба scalar256 explain-контроля воспроизводят native argmax. Проверены 32 исходных и 32 экспериментальных binary hashes, фактически загруженные библиотеки, общий header, исходники и build inputs. GPU освобождены.

Кандидат остаётся диагностическим. N=1 прошёл четыре prompt до проверенного лимита, но его target logits также численно расходятся после cache-границы. Общую совместимость, перенос в CUDA и ускорение DSpark закрывать нельзя.

После review снять первую расходящуюся границу на входах 231/232 explain, сохранив новые SHA и правильные batch columns. При одинаковых предшествующих данных сравнить Q, видимые K/V, полную маску, физические K/V размеры и FA parameters перед/после абсолютной позиции 256; проверить query-specific padding/attention отдельно от state rollback. После локального исправления повторить полный256 replay и свободные off/N=1/N=3 проверки. Производственную стоимость оценивать после совпадения ответов и без callback-дублирования.

Источники: [integrity](stage17/integrity.json), [проверяющий скрипт](stage17/verify-integrity.py), [воспроизведение](stage17/README.md). Большие бинарники, logits, raw probabilities, server logs и исходный generated server-context.cpp остаются локально. Этап 17 не закоммичен.
