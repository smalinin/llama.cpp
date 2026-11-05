# База знаний по оптимизации llama.cpp runtime

## Новое оборудование: 2026-11-05

Материнская плата, CPU, RAM и PCIe-порядок GPU заменены. Скорости, CPU/PCIe-калибровки и допуски памяти прежних серий относятся к старой машине и не являются baseline для текущей. Исторические отчёты сохранены. [Новая независимая серия Maya/наш runtime](experiments/1017-maya-newhost-20261105/REPORT.md) фиксирует новый hardware fingerprint и отдельные результаты.

Снимок знаний: 2026-11-03, `my_maya`, runtime `0e6f3cec3` (block-local default ON, pooling остаётся OFF). База хранится в этой ветке вместе с исходниками. Это начальная систематизированная выборка; полный исторический архив шире, чем каталог механизмов. Основной пример - DeepSeek-V4.1-Flash-1017/Q3; применимость к другой модели нужно установить по её архитектуре и фактическому backend path.

## С чего начать

| Задача | Документ |
| --- | --- |
| Оптимизировать другую модель | [HowTo](HOWTO.md), затем [паспорт модели](templates/MODEL.md) |
| Найти применимый механизм и его ограничения | [Каталог оптимизаций](CATALOG.md) |
| Проверить, что уже пробовали и почему отклонили | [Уроки экспериментов](LESSONS.md) |
| Посмотреть проверенную конфигурацию DeepSeek | [Паспорт DS41-1017](models/deepseek41-1017.md) |
| Оформить новый опыт | [Шаблон эксперимента](templates/EXPERIMENT.md) |
| Найти исходные доказательства и harness | [Локальный архив](evidence/README.md) |
| Подключить workflow агенту | [Skill](../../skills/runtime-optimize/SKILL.md) |
| Запустить текущие возможности DS41 | [readme_dsv41.md](../../readme_dsv41.md) |

База различает механизм, результат отдельного опыта и решение о включении по умолчанию. Проценты из разных опытов не складываются. Старый report описывает свой build, а актуальные defaults проверяются в исходниках текущего commit и в README. Исходные snapshots и hashes в evidence не переписываются при появлении новых выводов.

## Как пополнять

После опыта заполнить шаблон, добавить короткую запись в каталог или уроки, указать границы поддержки и source/build identity. Для новой модели создать свой паспорт в `models/`. Полные большие трассы и веса хранить в отдельном архиве; в Git сохранять выводы, нужные fixtures/harness, manifests, контрольные суммы и доступный locator. Локальная база уже содержит выбранные отчёты и источник проверок M2, поэтому их чтение не зависит от прежней папки проекта.

Новый [residual prefill profile после M2](experiments/1017-m2-residual-profile-20261103/REPORT.md) завершён: MoE составляет 54-57% суммы времени kernels, а cross-GPU traffic требует отдельного аудита. Узкий alloc input-copy wait не подтвердился bottleneck. [Следующий этап](experiments/1017-m2-large-context-20261103/REPORT.md) закрыл actual M2 ctx176128:172032+16, minfree601.25MiB, exact continuation и exit0. Read-only аудит подтвердил32 пары повторных CUDA API requests для HC root/reshape,1.342GB запрошенных повторно bytes; физический DMA и speed gain ещё не измерены. HC destination view sharing затем реализован и проверен, см. новый отчёт ниже. Dense indexer traffic и свежее matched сравнение с Maya остаются отдельно. Q2/Q4 routed MMQ остаются отдельными будущими проверками. Здесь они не помечены выполненными.

[HC copy-sharing design и historical cost screen](experiments/1017-hc-copy-design-20261103/REPORT.md): предыдущие результаты закоммичены9b74cc6e6. Подготовлен opt-in same-split prototype; allocator view bookkeeping и lifetime контракт описаны в [design](designs/1017-hc-copy-sharing.md), реализован и проверен следующим этапом. Historical CUPTI подтвердил staged D2H/H2D records и пары перед одним destination launch; gain в том historical screen не измерялся; результат реализации ниже.

[HC copy-sharing: реализация, checks и default ON](experiments/1017-hc-copy-sharing-20261103/REPORT.md): PP512/8K/98K +4.06/+4.26/+3.77% к bracketingOFF, TG -0.50..+0.37%, combined GPU reserve -640MiB.36 exact target/draft groups,36rollback,2failure/retry,deltaNLL0,text/code;27server outputs/counters exact. По отдельному запросу пользователя unset/1 включает, GGML_SCHED_COPY_RESHAPE=0 отключает. Guards ограничивают sharing; другие models/KV/media/parallel остаются отдельными gates.


## HC default ON:176128 и следующий indexer шаг

[Свежий реальный gate и indexer audit](experiments/1017-hc-long-indexer-20261103/REPORT.md): ctx176128/172032+16 прошёл с sampled minimum761.25MiB>=512,actual15360GPU bundles/placement exact,exit0.8K/172K continuation exact. Новый separate ctx100352 profile подтвердил source20 ratio1, block8/top2048blocks и72 full-score CUDA2->3/72 full-mask CUDA3->4 transfers на98K. Предлагается [placement-only перенос CONT/padding/pool](designs/1017-indexer-placement.md); на момент аудита runtime этого шага ещё не был реализован; результаты O11 приведены ниже. Selected copy correlation имеет metadata/size/consumer scope; Nsight graph-ID/incomplete-event diagnostics не позволяют объявлять полную critical-path decomposition. Matched Maya и Q2/Q4 остаются далее.


## Producer-local candidate pool: opt-in OFF

[Новый этап](experiments/1017-candidate-pool-local-20261103/REPORT.md) реализован после initial commitc3c91dda8: CONT/padding/pool у score producer, selector на прежней GPU. На fixed Q3/Q8KV/BF16K3 98K PP+1.46%,8K-0.28%; TG gain не установлен (98K-1.07%,singleton). No_alloc combined compute reserve -593MiB@100K/-1073MiB@176K. Actual176128/172032+16 min761.25MiB,exact placement/IDs,exit0.48 precision groups двух фаз exact,deltaNLL0; corrected operator80cases и isolated fixed32capture/replay прошли. Wide full-score copy заменена134block-score requests; tiny verification3/4queries остаётся на fallback. Флаг LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 только opt-in, defaultOFF; применять перед загрузкой контекста. Результаты и ошибки стендов сохранены, дальнейшее продвижение требует review.


## Block keep transport: реализован, guarded default ON

[Согласованный дизайн](designs/1017-candidate-block-local.md) реализован и [проверен](experiments/1017-candidate-block-local-20261103/REPORT.md). Existing F32 block keep переносится между GPU, прежний REPEAT разворачивает маску локально; scheduler events/copy slots прежние. По запросу пользователя unset/1разрешает LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL,0отключает; значение фиксируется до загрузки контекста. Producer-local pooling остаётся отдельным opt-in/defaultOFF.

В основных gates pooling1 фиксирован на всех arms: PP98K1121.83->1154.72t/s (+2.93% относительно pool1/block0), TG46.23->45.94 (-0.62%,singleton; gain не установлен).8K/512 candidate-ветвь неактивна, небольшие deltas не считать ускорением.24 raw target/draft groups/two98K corpora exact,deltaNLL0;4seed failures/retry,operator98cases/fixed32capture и external compute failure/retry passed. No_alloc -520.52MiB combined compute@176K;actual176128/172032+16/repeat8K min1143.25MiB,15360GPU bundles/IDs/routes exact,exit0.134 wide fullmask3->4 requests заменены134compactkeep,26.700GB->3.338GB logical API payload,slots0..3; это неphysicalDMA и не8xengine gain. По отдельному запросу пользователя BLOCK_LOCAL promoted defaultON;pooling defaultOFF прежний. Далее matched Maya и Q2/Q4; pool0 native на момент той серии ещё не был закрыт; новый gate приведён ниже. Media/parallel>1/auto-fit262K остаются отдельно.

[Promotion checks](experiments/1017-candidate-block-local-20261103/REPORT.md#default-on-по-запросу-пользователя):7fresh no_alloc процессов/default==explicit1 приpool0иpool1;0fallback. Это не новый native speed gate дляpool0.


## Текущие candidate defaults: native gate закрыт

[Отчёт](experiments/1017-pool0-default-20261103/REPORT.md): runtime `my_maya 0e6f3cec3` и defaults не менялись. Pooling не задан (OFF); в default arm BLOCK_LOCAL также не задан (ON). На двух 98K text/code корпусах 24 группы logits/taps/state/rollback совпали побитово, delta NLL=0; четыре seed failures/retry прошли. Native OFF/default/OFF: 27 output arrays и draft counters совпали. PP98K 1107.75 -> 1138.16 t/s (+2.75%); TG 46.21 -> 45.82 (-0.84%, по одному измерению на arm, ускорение TG не установлено).

Actual ctx=163840 / prefix=159744 / repeat 8K прошёл: минимум 679.25 MiB свободной VRAM, все 15360 экспертных bundles на GPU, placement совпал, exit 0. Context=176128 не допущен расчётом при pooling OFF; прежний actual 176128 с pooling=1 остаётся отдельной серией. Это fixed Q3/MXFP8 Engram/Q8 KV/BF16 DSpark K3/P1 конфигурация, не универсальный предел контекста.

Ошибка post-test сборки сводки (`queue.py` перекрыл stdlib) вызвала повтор metadata. Исходные precision/native/payload manifests сохранены побайтно; память и routes metadata сверены с исходными admission inputs. Incomplete manifest, старые hashes и recovery diagnostics сохранены в отчёте. Runner переименован и защищён от выполнения при импорте.

Далее: matched Maya comparison, свежий residual profile, затем Q2/Q4 MMQ. Media, parallel>1 и auto-fit 262K остаются отдельными проверками. Нового commit/deploy нет.


## Новая платформа: подробная телеметрия Maya и нашего runtime

[Отчёт](experiments/1017-maya-telemetry-20261105/REPORT.md): на записанном prefill8K условное число вычисляющих GPU Maya3.33, наше1.32;98K3.93/1.36. Polling/transport исключены, это не SM occupancy. Maya исполняет все40слоёв, наш pipeline enabled, но фактическое перекрытие слабее. TG profile указывает на target fragmentation и guard: anchor+5draft даёт6queries, KV-wide допускает6, обычный float/MoE batch guard5. Это кандидаты следующих native опытов, не реализованный фикс.

Профилировщик снижает наш TG примерно на27-29%, есть CUPTI GraphId diagnostics; CPU sampling/OSRT недоступны. Maya98K восстановлен после STOP/controller timeout без post-control; original manifest остаётся incomplete. Полный8K и оба наших захвата сохранены, runtime/defaults прежние. Нельзя выдавать recorded idle/API time за полную native critical-path decomposition. Следующий приоритет: bounded chunk/input/copy/feature lifetime для PP, counters-only sizes1..6 для verification.

## Independent CPU lookup overlap on the new host

[Approved prototype and gates](experiments/1017-cpu-lookup-overlap-20261105/REPORT.md): CPU GET_ROWS can start before a preceding CUDA fence when existing protected staging/slot ownership proves isolation. Initially tested as opt-in; now guarded default ON by user request. Unset/1 enables and `GGML_SCHED_CPU_LOOKUP_EARLY=0` disables. See the report promotion section for the rebuilt library and default checks. New-host native OFF/ON/OFF gives PP8K +6.56%, PP98K +5.34%, PP512 unchanged within variation. All 24 output arrays/draft counters and 12 full-model precision groups match exactly; delta NLL=0 on both corpora, four seed failure/retries and 42 scheduler cases passed. The native sampled floor is 1271.25 MiB. Promotion changes only the startup default; measured speed remains from the explicit-ON experiment.

This is a narrow prefill gain, not the full Maya gap. Next: residual prefill dependencies with the prototype enabled, then the separate six-query float/MoE verification path. Prior-platform timings remain historical only.


## Residual prefill on the new host

[Host/API attribution and negative queue/slot screens](experiments/1017-prefill-residual-20261105/REPORT.md): CPU lookup stays default ON at runtime f796e544f. Queue4x reduces 8K launch-call host time3.198s->0.143s but PP stays1431/1427t/s; waiting moves to protected slot retirement and target drain. Private copies8 with queue4x gives1419t/s, actual minimum625.25MiB, no useful gain. Neither setting is adopted. First five phases have29 exact same-prefix output/counter requests and clean exits; the copies8 controller's post-test log-format assertion is recovered separately, original incomplete manifest retained. The existing-CLI shared gate/up6/29 co-location screen removed24 backward transitions but PP changed-0.65%; it is not adopted. No scheduler reorder is implied.


## KV roots and reader-side deferral: audit complete, implementation pending

[Ownership audit](experiments/1017-kv-owner-audit-20261105/REPORT.md) confirms real source20 replica SET_ROWS and a CUDA2->3->4->2->3->4->5->2 submission detour. Source20 rotation/Hadamard physically executes onCUDA4 after eager LID writes; layer number alone does not prove producer device.374 graph snapshots, nine selected target chunks,36source20 writes,42 external scheduler cases and five exact default8K requests passed, actual minimum1323.25MiB, exit0. Armed metadata capture costs17.41% PP and is not a speed baseline.

Existing decode-only LLAMA_DSV41_DEFER_CACHE_UPDATES is a possible reuse point. [Proposed prefill design](designs/1017-prefill-cache-deferral.md) preserves ops/events/slots and must flush every pending replica on KV-only early return. User approval is obtained; the [opt-in prefill prototype](experiments/1017-prefill-cache-deferral-20261105/REPORT.md) passes precision/state/rollback and actual ownership/KV-only flush checks. Clean native OFF/ON/OFF completed at identical ctx92160 (PP8K+4.45%, PP88K+7.40%, PP512-4.74%, native floor535.25MiB); default remains OFF and production modules/presets unchanged. The source-level Maya reader-side pull/ack protocol is an architectural reference, not a measured gain or imported numerics.


## Approved prefill KV deferral: exact but memory-costly opt-in

[Completed evidence](experiments/1017-prefill-cache-deferral-20261105/REPORT.md): same private module with flags0/1/0, ctx92160/longprefix90112, new-host fixed Q3/MXFP8 Engram/Q8 KV/DSparkK5/full/pmin0.5/U512/P1/15360GPU experts. PP512724.48->690.17t/s (-4.74%), PP8K1412.14->1474.92 (+4.45%), PP88K1449.66->1556.94 (+7.40%); TG gain not established. All27native output arrays/draft counters,12paired precision groups/two88K corpora/NLLdelta0/four failure-retries exact. Actual source20 producerCUDA4->CUDA2, pending KV-only flush and419raw/CSA/LID writes per arm verified. CUDA5 joint reserve+795.02MiB, native minimum535.25MiB; higherctx100352..94208 failed conservative admission. Flag LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1 opts in; unset/0 OFF. Code available for review, production modules/presets unchanged; no new commit/default promotion. Do not infer broad auto-fit262K/F16/media/parallel/CPU-cache support. Workspace and short-prompt regression are separate follow-up candidates; six-query verification remains independent.


## New-host verification width screen: six-query fragmentation measured

[Completed screen](experiments/1017-verify-width-screen-20261105/REPORT.md), after runtime commit3854c3785: fixed prefix/tokens/state/n_rs_seq5, raw target decode+sync medians at8K:5queries43.03ms,6queries68.55ms (+59.30%);88K46.02->68.17ms (+48.12%). Separate lightweight counters show MoE120->720 and float275->950 matmuls, backend graphs44->114 /43->113. All156per-arm logits/splits match native/counter;24sampled state checkpoints per arm and base restores exact. Actual floor1391.125MiB, exits0, source/modules unchanged. Cold smaller-width graph instantiation rows are retained; measured width6has0instantiations and all llama graphs reuse. This is not total server TG/acceptance or a new implementation/default. [Next bounded plan](designs/1017-six-query-moe-float.md): isolate grouped MoE6, then floating sample-grid6; shared-weight MMVF6 is a later separate kernel candidate.
