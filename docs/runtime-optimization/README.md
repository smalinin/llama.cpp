# База знаний по оптимизации llama.cpp runtime

Снимок знаний: 2026-11-03, `my_maya`, runtime `c3c91dda8` плюс проверенный candidate-pool opt-in delta (пока без нового коммита). База хранится в этой ветке вместе с исходниками. Это начальная систематизированная выборка; полный исторический архив шире, чем каталог механизмов. Основной пример - DeepSeek-V4.1-Flash-1017/Q3; применимость к другой модели нужно установить по её архитектуре и фактическому backend path.

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

[Свежий реальный gate и indexer audit](experiments/1017-hc-long-indexer-20261103/REPORT.md): ctx176128/172032+16 прошёл с sampled minimum761.25MiB>=512,actual15360GPU bundles/placement exact,exit0.8K/172K continuation exact. Новый separate ctx100352 profile подтвердил source20 ratio1, block8/top2048blocks и72 full-score CUDA2->3/72 full-mask CUDA3->4 transfers на98K. Предлагается [placement-only перенос CONT/padding/pool](designs/1017-indexer-placement.md); runtime этого шага ещё не реализован. Selected copy correlation имеет metadata/size/consumer scope; Nsight graph-ID/incomplete-event diagnostics не позволяют объявлять полную critical-path decomposition. Matched Maya и Q2/Q4 остаются далее.


## Producer-local candidate pool: opt-in OFF

[Новый этап](experiments/1017-candidate-pool-local-20261103/REPORT.md) реализован после initial commitc3c91dda8: CONT/padding/pool у score producer, selector на прежней GPU. На fixed Q3/Q8KV/BF16K3 98K PP+1.46%,8K-0.28%; TG gain не установлен (98K-1.07%,singleton). No_alloc combined compute reserve -593MiB@100K/-1073MiB@176K. Actual176128/172032+16 min761.25MiB,exact placement/IDs,exit0.48 precision groups двух фаз exact,deltaNLL0; corrected operator80cases и isolated fixed32capture/replay прошли. Wide full-score copy заменена134block-score requests; tiny verification3/4queries остаётся на fallback. Флаг LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 только opt-in, defaultOFF; применять перед загрузкой контекста. Результаты и ошибки стендов сохранены, дальнейшее продвижение требует review.
