# Этап 8: Q4_K sidecar и точность CUDA для DeepSeek-V4.1

Q4_K sidecar проверен: на четырёх запросах ускорения относительно MXFP4 нет, выбранные токены и текст совпали. Дополнительно найден и исправлен обход `GGML_PREC_F32` в CUDA MUL_MAT. FP32 hyper-connections DeepSeek-V4.1 проверены отдельно и оставлены экспериментом, без изменения рабочего графа. Установленный сервер и рабочие профили не менялись. Коммита нет.

Дата: 8 октября 2026 года. Ветка `my_build_upstream_20261007`, HEAD `11638b68545860e96b055798e995bb14be3d0e88`.

## Q4_K sidecar

Исходный `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf` уже квантован: девять крупных expert tensors имеют тип MXFP4, attention/projection преимущественно Q8_0, остальные тензоры F32/BF16. Создана отдельная экспериментальная копия `DeepSeek-V4.1-Flash-DSpark-Q4_K-from-MXFP4.gguf`. Только девять MXFP4 tensors преобразованы в Q4_K; все остальные 69 tensors проверены SHA256 и побайтно совпадают. Tokenizer, model metadata и логические размеры tensors сохранены. GGUF writer убрал конечную единичную размерность у `conf_proj.weight`; логическая форма `[5376,1,1,1]` не изменилась.

Это повторное квантование уже квантованных весов, без восстановления исходной точности. Оно проверяет данный переход MXFP4 -> Q4_K и работу соответствующих CUDA kernels. Контроль с GGUF, независимо полученным из исходного checkpoint в другой точности, не выполнен.

Размер увеличился с 7598,60 до 8003,60 MiB: **+405 MiB**. Модель оставлена в локальном каталоге артефактов и не включается в Git. Исходные модели не изменялись.

На Hugging Face найден [Lucebox Q2_K/Q4_K draft](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF). Прочитан только заголовок, 8 MiB из 4 923 041 440 bytes, revision `11403d34ca19eb1ac2b532b5a05e61a4b0c61485`. Архитектура `deepseek41-dflash-draft`, metadata и часть имён tensors отличаются от поддерживаемого здесь `dflash`. Этот файл не загружался в llama.cpp и не использовался в benchmark. [Результат проверки](stage8/lucebox-q2k-q4k-inspection.json).

## Сравнение MXFP4 и Q4_K

Использован неизменный snapshot этапа 5. Исправления FP32 этого этапа и отдельное исправление Continue в этот benchmark **не входят**. CLI отличается только портом и путём draft; hashes загруженных библиотек совпадают. Условия повторяют этап 7: target layer split `1,1,1,1,1,0.4`, fit off, 8192 context, batch 2048, ubatch 512, один slot, F16 KV, 12 CPU threads, draft 3, confidence 0, temperature 0, seed 1234, `cache_prompt=false`.

Два запуска по 18 запросов: два diagnostic, четыре warmup, три серии измерений по четырём prompt. Всего **36/36 completion requests прошли**, оба сервера завершились с кодом 0. Внутри каждого варианта все три измеренных повтора совпали по token IDs, тексту и draft counters. Между MXFP4 и Q4_K token IDs и текст также совпали для всех четырёх prompt. Сравниваются одинаковые выходы, длина 120/256/256/256 tokens. Полнота и качество ответов отдельно не оценивались.

| Prompt | MXFP4, tok/s | Q4_K, tok/s | Изменение Q4_K |
| --- | ---: | ---: | ---: |
| Исходный | 42,85 | 42,39 | -1,08% |
| HTML/SVG pelican | 45,03 | 44,99 | -0,10% |
| Python merge intervals | 42,69 | 41,64 | -2,46% |
| Database indexes | 37,41 | 36,95 | -1,23% |

Показана медиана трёх измерений; warmup исключён. Это две последовательные серии Q4_K, затем MXFP4, без чередования процессов. Малые отличия скорости не устанавливают преимущество одного CUDA kernel, но ни один prompt не показал ускорения Q4_K.

Acceptance MXFP4 -> Q4_K: исходный prompt 26,77% -> 25,87%; SVG 27,14% -> 27,14%; Python 24,37% -> 22,84%; indexes 17,00% -> 16,12%. Concurrent peak VRAM 211060 -> 211456 MiB, +396 MiB. Оставить MXFP4 для проверенной сборки; меньший размер и ускорение от перехода на Q4_K не подтверждены. [Сводка](stage8/q4-summary.json), [точные модели и hashes](stage8/q4-comparison-manifest.json), [69 сохранённых tensors](stage8/q4-preserved-tensors.json).

## Локализация CUDA precision

Первый диагностический callback сохранял промежуточные tensors при decode с позиции 32, включая контроль с одинаковой single-token KV history перед переходом на batch 2/4. Callback оставил width 1 побайтно прежним, но изменил logits width 2/4 относительно сохранённого server-matched replay этапа 7; у width 2 изменился argmax на позиции 59. Поэтому этот capture не используется как неизменённый baseline или самостоятельное доказательство причины всего расхождения.

Capture указал на BF16 router layer 0: одинаковый F32 вход, но результат меняется при переходе от width 1 к width 2. Выполнен отдельный replay реального `blk.0.ffn_gate_inp.weight` `[5120,384]` и одного сохранённого входа. Во все столбцы записан один и тот же вектор; CPU reference вычислен с double accumulation по точным BF16 weights и F32 input. Эксперимент не использует модель, cache или DSpark.

При width 1 CUDA MMVF сохраняет F32 input; при width 2+ MMF округляет его в BF16 для tensor cores. MMF выбирался даже при `GGML_PREC_F32`. Для F32 weights width 4+ также выбирался MMF с TF32. Дополнительно cuBLAS handle по умолчанию разрешал TF32 даже на fallback с запрошенным FP32.

На реальном BF16 router max error против CPU: width 1 около `6.28e-7`, width 2+ до исправления `0.0024637`, после `1.16e-6`. Максимальное отличие от scalar CUDA уменьшилось с `0.00246382` до `1.43e-6`. Обычный режим precision оставил все показатели изолированного replay прежними. [Replay до](stage8/router-replay-before.jsonl), [после](stage8/router-replay-after.jsonl).

## Изменения исходников

- `ggml/src/ggml-cuda/ggml-cuda.cu`: MUL_MAT с `GGML_PREC_F32` обходит MMF, округляющий operands; cuBLAS для этого режима временно использует `CUBLAS_DEFAULT_MATH`, затем восстанавливает штатный TF32 mode handle.
- `tests/test-backend-ops.cpp`: существующий `test_mul_mat` получил параметр precision; добавлены четыре F32 regression cases с width 1/2/4/16 и строгим CPU comparison.

CUDA Release build `llama-server` и `test-backend-ops` прошёл. До исправления два из четырёх новых cases падали, после **4/4 прошли**. На Ada RTX 4090 и Ampere RTX 3090 прошли по 70 default float matmul checks и четырём FP32 cases, **148/148**. BF16 отдельно проверен replay с точным CPU reference: обычный CPU backend сам округляет input в BF16 и не подходит как строгий FP32 reference для такого теста. Первоначальные BF16/F16 cases с неверно выбранным CPU reference исключены; логи сохранены локально.

Изменение затрагивает все CUDA MUL_MAT nodes, явно запрашивающие FP32, поэтому может менять ответы и стоимость вычислений относительно прежнего пути с TF32/BF16 rounding. Качество модели и полный набор других архитектур после этого изменения не оценивались. Исправление не является доказанным ускорением DSpark.

## Полная модель

Контроль только исправленного router precision убрал argmax divergence width 2 на 65 teacher-forced позициях, но width 4 по-прежнему расходился на позиции 33. Численные отличия logits остаются большими: max abs width 2/4 около 5,28/7,10, RMS 0,244/0,259. Это не устранение общего расхождения logits. Данные: [router-only replay](stage8/router-only-summary.json).

Дополнительный диагностический вариант задавал FP32 для `hc_mixes` через callback. В нём argmax совпал на всех 65 позициях для widths 1/2/4. Затем флаг временно установлен прямо в графе: replay без callback побайтно совпал с прототипом для каждого width, что исключает влияние callback на этот результат. Logits между widths не стали одинаковыми: max abs width 2/4 против width 1 около 3,68/2,65, RMS около 0,204/0,221. [Сводка HC](stage8/hc-prototype-summary.json), [проверка без callback](stage8/final-vs-hc-prototype.json).

Во всех teacher-forced контролях используется один и тот же старый native prefix этапа 7. Новые precision settings меняют исходное распределение, поэтому совпадение argmax на этом prefix не гарантирует совпадение обычной генерации, создающей новую историю.

Это подтверждено реальными запросами: **6/6 completions прошли**, по три greedy повтора DSpark off/on. Внутри каждого режима ответы совпадают. Оба режима завершились естественным EOS после 95 tokens, но off/on различаются с индекса 23, то есть с **24-го выходного токена**. Медиана двух прогретых diagnostic запросов: off 45,26 tok/s, DSpark 3 41,26 tok/s; ответы разные, поэтому это не сравнение одинаковой работы и не доказанное ускорение. [Результат HTTP контроля](stage8/final-greedy-summary.json).

Поскольку свободная greedy generation не стала совместимой, новый флаг FP32 HC **убран из рабочего графа**. В `src/models/deepseek4.cpp` итоговых изменений нет. Воспроизводимый HC эксперимент и его результаты сохранены отдельно. Для review остаются два изменённых исходника: CUDA precision и существующие backend tests. Источник остаточного расхождения не установлен; следующий шаг требует отдельной диагностики на новой одинаковой истории, а не дальнейшего выбора quants вслепую.

## Артефакты и границы результата

Большие logits dumps, tensor captures, GGUF, бинарники и полные журналы остаются в `/home/sergei/_my_sync/llama_upstream_review/stage8/`. Snapshot `candidate-bin` содержит оставленное для review исправление CUDA precision; `final-bin` содержит отвергнутый HC experiment с этим исправлением и не является рекомендуемой рабочей сборкой. Оба включают ранее подготовленный отдельный Continue fix, который к результатам target-only DeepSeek не относится. Snapshot этапа 5 и установленный `/home/sergei/_llama_cpp/my_build/llama-server` не менялись.

Серии measured requests Q4_K/MXFP4 не пересекались с GPU precision controls. Первые unit controls во время загрузки MXFP4 сервера завершились до его diagnostic и measurement requests. CPU compilation, выполнявшаяся во время benchmark, завершилась до measured requests соответствующей серии; cold diagnostics и warmups исключены из сравнения.

Строгая совместимость всех logits и устойчивая выгода DSpark остаются открытыми критериями. Результат передать на review перед коммитом; commit и deploy не выполнялись.
