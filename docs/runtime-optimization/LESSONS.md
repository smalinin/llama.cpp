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


## Candidate pooling: проверять фактическую маршрутизацию и startup reserve

[Опыт](experiments/1017-candidate-pool-local-20261103/REPORT.md) подтвердил134 wide block-score transfers вместо full score и отдельный native PP+1.46%, не8xengine gain. Смена arithmetic не понадобилась; first post-pool ADD пришлось привязать к прежнему selector owner, иначе scheduler мог утащить всю ветку вслед за pool.

- Backend hints не гарантируют маршрутизацию маленького стенда: set_tensor_backend делает scheduler non-reset, а implicit graph_compute allocation может сбросить hints. Нужен explicit alloc_graph после assignments и проверка actual backend IDs/counters. Первый same-GPU ложноположительный run отвергнут; corrected80cases/fixed32 validated.
- У arbitrary reshape ne1 может быть n_blocks, а не query count. В pointer audit отдельно классифицированы wide prefill и fallback3/4 по canonical score shape; пять tiny full-score transfers не являются регрессией wide пути.
- Dynamic OFF->ON в уже живом контексте удержал расширенный scratch и дал shutdown size warning. FreshON constructor reserve1206.26MiB покрывает retained1197.96MiB и проходит без warning; production startup и diagnostic mode switching проверяются отдельно. Это не разрешение игнорировать warning обычного сервера.
- Post-child JSON path error не отменяет nativeexit0, но raw complete:false нельзя переписывать. Сохранены measured runner/manifest/error и независимый validator фактического12-group output.
- No_alloc reserve, actual minfree и usable context различаются: combined -1073MiB при176K не поднял global floor761.25MiB, который ограничивает другая GPU. Для262K нужен отдельный admission/actual gate.
- Guard tightening после precision записан отдельным patch/hashes. Для неизменившейся40layer ветки повторены final metadata/native/memory/copy gates; old precision identity не выдана за final binary measurement.


## Передавать компактное представление до разворачивания

[O12](experiments/1017-candidate-block-local-20261103/REPORT.md) переносит существующий F32 block keep и повторяет прежний REPEAT у consumer. Wide logical API payload уменьшился8x, а PP98K вырос только2.93% в своей серии: bytes не равны critical-path time. Сначала установить numerical equivalence и source/query ownership, затем сохранить allocator lifetime/copy slots и проверить полный src/dst range/shape/stride/consumer. Shared key owner не равен владельцу всех query projections. Graph-local pointer map не заменяет явные graph dependencies и reuse key.

Compute reserve может одновременно вырасти на producer-side consumer (здесьCUDA3+344MiB) и уменьшиться на другом (CUDA4-864.52MiB). Combined savings не распределяются поровну и не предсказывают global floor. Initial metadata с concurrent operator дали несопоставимые free-before; повтор idle metadata обязателен для admission. Pool0metadata/fixed operator и реальный pool1native - разные доказательства. Короткий8K путь не входит в candidate branch, поэтому его небольшую delta нельзя приписывать новому механизму.


## Проверять defaults нужно и в самом стенде

[Pool0 native gate](experiments/1017-pool0-default-20261103/REPORT.md): убрать переменную из launcher недостаточно, если helper задаёт её внутри corpus loop. Проверять child environment, source и actual graph owners. Значение, фиксируемое конструктором, сравнивать в свежих процессах. Успех memory gate при pooling=1 не переносится на pooling OFF: меняется per-device reserve; допуск считать по каждой GPU и подтверждать actual запуском.

Не давать исполняемым Python harness имена stdlib modules (`queue.py`). Защищать запуск `if __name__ == '__main__'`. Повтор metadata при сборке итогов может перезаписать manifests: сохранить диагностику и исходные hashes, проверить независимые admission inputs, отдельно назвать невосстановимые bytes. Не исправлять raw incomplete manifest на complete. Near-prefix другого размера без token oracle является memory/lifecycle gate, не доказательством качества.


## Телеметрия CUDA и асинхронного pipeline на новой машине

[Опыт 2026-11-05](experiments/1017-maya-telemetry-20261105/REPORT.md):

- Отделять GPU polling/transport kernels от compute. Maya xfer_get/wait_flags увеличивают сырую concurrency; NVML load и число активных kernels не равны SM occupancy.
- Enabled pipeline flag не доказывает overlap. Измерять интервалы по GPU, copy-slot lifetime и реальное продвижение chunks. Maya PP8K имеет условную compute concurrency3.33 против1.32 у нас, даже без upper-layer skip у Maya.
- n-max задаёт draft tokens; anchor+5 означает6queries. Guards KV, float matmul и MoE могут различаться. Их безопасное расширение требует отдельных численных/state/rollback gates.
- Node-level CUPTI может сильно менять TG и page-fault profile. Сохранять native speed отдельно, pre/profile/post controls, diagnostics и sampled-stack limitations. Большой поток Instantiate под Nsight сначала подтвердить counters-only в native.
- CPU_STATS из другого tensor path может печатать0/NaN без охвата реально исполняемого GGUF IQ. Проверять, где стоят timers; ноль не означает отсутствие работы.
- Сначала сохранять ответ и counters, потом ждать profiler STOP/flush. Большая98K трасса превысила120s, поэтому controller failure не равен GPU failure; incomplete manifest и recovered evidence сохраняются отдельно.

## A long host wait is not all removable work

[CPU lookup overlap](experiments/1017-cpu-lookup-overlap-20261105/REPORT.md) measured 3.00 s / 41.00 s of previous-GPU transition waiting at 8K/98K, while CPU lookup itself took 0.21 s / 2.62 s. Running that lookup during GPU work improved native PP by 6.56% / 5.34%; the GPU computation inside the inclusive wait still had to run. Do not turn sampled event-sync percentages into promised speedups.

Zero cross-backend inputs does not by itself prove independence. CPU scratch was reused by three lookup groups. A safe early start required immutable source weights, owned row IDs, no direct GPU read of host scratch, protected per-copy host stages, and isolation of the previous graph as well as the current graph. Failed/unsafe predecessors retain conservative ordering. The existing async-input API already rejected shared backend buffer types; an extra duplicate guard was unnecessary.

Separate native timing from raw-logit/state tests and preserve abandoned attempts. This run retained a deliberately cancelled loader and a wrong negative-fixture expectation, then reran the final tests with source/library hashes identical to the measured native candidate. Exact OFF/ON logits and state are the correct gate for transport-only changes; the general NLL quality tolerance does not justify a scheduling mismatch.


## Enqueue latency can move without accelerating inference

[Residual prefill](experiments/1017-prefill-residual-20261105/REPORT.md) covered the actual CUDA13 `__cudaLaunchKernel` ABI after detecting an initial coverage hole. Match imported symbols to the active toolchain before attributing unexplained host time. Record failed builds and partial coverage rather than treating the unexplained remainder as CPU math.

Queue4x cut host launch duration about22x without a useful native PP gain. Protected input-copy reuse and final target drain absorbed the wait; private copies8 then moved some waiting back into enqueue calls, again without faster PP. Preserve actual memory floor, source/library identity and same-process pre/post controls. A faster host API is not automatically a faster engine or GPU kernel.

Names ending `(view)` do not prove a metadata-only split. Here `cache_k_l30 (view)` is SET_ROWS and writes KV. First/last node summaries omit full root ownership and allocator reuse intervals; do not remove cross-GPU fences based on names or pointer equality across graphs. Two real shared-expert backward paths are placement candidates for an existing-CLI screen, not evidence of redundant synchronization.

A final log-format assertion can fail after successful requests and exit0. Keep the original incomplete manifest immutable and provide separately hashed recovery checks; do not rewrite it to pass. Screens with16 output tokens establish neither sustained TG nor full logits/state/rollback correctness.

The shared-expert placement screen removed all24 observed backward transitions at boundaries6/29 and cut launch count51068->49634, yet native PP changed-0.65%. Confirming activation does not establish a useful engine gain. The retained KV SET_ROWS path is real work; the next audit must establish writer/root-buffer/slot ownership before any scheduling change.
