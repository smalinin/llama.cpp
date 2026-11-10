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


## Logical source layer is not actual producer ownership

[KV owner audit](experiments/1017-kv-owner-audit-20261105/REPORT.md) found source20 ROPE/Hadamard running onCUDA4 after LID replica fanout, although model.dev_layer(20) isCUDA2. Its F32 output feeds Q8_0 cache roots onCUDA2/3/4/5, then execution returns toCUDA2. Use actual operand/root buffer devices and exact within-ubatch copy lineage. A split containing a view can still perform SET_ROWS; source/destination/cache/input-stage lifetimes differ.

Existing scheduler getter sched_get_tensor_backend uses find_or_insert, so an observer should not call it on arbitrary untracked operands. Public buffer-type device metadata avoids that mutation. Full graph metadata can slow an armed request materially even without eval callbacks; keep native pre/post separate.

Reusing decode's deferred replica writes for prefill needs a KV-only flush: the truncated prefix can return at layer20 without later reader layers, but their persistent caches still need every prefix row. The empty-map assertion is a correctness check, not an obstacle to delete. Preserving the same slot/event contracts is required even when graph order changes. A reviewable [design](designs/1017-prefill-cache-deferral.md) is prepared; implementation and speed/precision gates are pending.


## Prefill replica deferral: correctness and placement gains have a workspace price

[Completed candidate](experiments/1017-prefill-cache-deferral-20261105/REPORT.md) keeps the same SET_ROWS and existing protected events/slots but moves remote writes to their readers, flushing all pending updates on KV-only exit. Actual source20 placement returns fromCUDA4 toCUDA2 and graph counts fall31->24 /50->40. Those counts are diagnostic; measured PP changes are-4.74% at512,+4.45% at8K,+7.40% at88K. It remains OFF, not a blanket deployment recommendation.

Cache root labels are not allocation identity: raw ring, CSA and LID may all be named cache_k_l20 but have widths512/512/128 and distinct allocations. Match root/storage data and size within the current ubatch, compare semantic shapes/types/counts across processes, and retain numeric state checks. All419writes per arm and first-reader ordering passed; graph label collisions initially fooled an auxiliary checker.

Joint target/draft/tap allocation can grow even when target compute reserve shrinks slightly. Here CUDA5 gains795.02MiB, mostly draft/taps, and actual native floor drops to535.25MiB; raw-logit checks alone observed801.25MiB and would overstate spare memory. Admit both arms at the same reduced context with fresh baselines; report metadata, raw precision, ownership diagnostics and native sampling separately. A missed opt-in environment variable produced an OFF ownership trace; activation checks caught it before native timing, and it is retained as baseline evidence rather than relabeled ON.


## Count the anchor and separate graph shape from draft policy

[New-host width screen](experiments/1017-verify-width-screen-20261105/REPORT.md): a draft cap5 creates6target queries including its anchor. KV-wide support6 did not mean the floating/MoE guard5 also accepted6. With identical prefix/proposal IDs and restored target state,5->6queries increases target-call median59.30% at8K and48.12% at88K. MoE dispatch120->720, float matmuls275->950; ordinary quantized matmuls remain331. Do not blame acceptance or prefill for this isolated existing branch.

All llama graphs were reused; measured six-query CUDA Graph instantiation was0. A reverse-pass first measured sample for widths1/2/3/5 still created CUDA Graph instances. Keep those rows and show pass medians instead of falsely labeling every measured sample hot. Lightweight counters varied median cost by about-0.68..+0.55% here, but this is still raw-logit target timing, not server GPU-sampler TG. State restore/feature bridge/SHA are outside the timer; differing prefix/K policies and sampler paths need independent end-to-end evidence.

For a bounded fix, isolate grouped MoE6 and floating sample-grid6 before changing shared-weight CUDA kernels. A global guard increase could enable unintended F16 compact/other paths and destroy attribution. Generic kernel availability or one exact greedy continuation is not a substitute for all-position logits/taps/state/rollback and actual joint memory gates.


## MUL_MAT and MUL_MAT_ID have different width selectors

The [direct6 MoE fixture](experiments/1017-verify-moe6-20261105/REPORT.md) rejects a blanket sixth-query grouping on Ampere Q3_K. Audit `get_mmvq_mmid_max_batch`, not the ordinary quantized MUL_MAT selector: Ampere Q3_K ID dispatch caps MMVQ at5, while Ada permits8. Raising an unrelated predicate can therefore change arithmetic on only one device of a mixed topology. Preserve dispatch-specific exactness evidence and failed cases.

The [5+1 follow-up](experiments/1017-verify-moe5plus1-20261105/REPORT.md) passes all64 relevant linear-operator cases yet fails an initial full-model serialized rollback-state comparison while recorded logits/next-draft assertions match. This is unresolved, not an accepted relaxation of the state contract or evidence of a speed gain. Diagnose the exact serializer components with fresh OFF/ON before changing arithmetic, rollback or test thresholds.

The subsequent instrumented OFF/ON state diagnostic passed, but the original-harness repeat diverged at first8K prefill before the new six-query branch can activate; three fresh OFF8K clear/reset repeats then matched the canonical reference. This does not establish a common cause or exonerate the candidate. Separate independent prefill variability from the proposed mechanism using common captured prefix state and preserved context/graph lifecycle. Keep every failure and do not equate an instrumentation-sensitive pass with a fixed bug.


## Freeze prefix values before attributing a verification difference

The [shared-prefix diagnostic](experiments/1017-verify-shared-prefix-20261105/REPORT.md) restores the same target/draft state in fresh startupOFF/ON/OFF contexts.117target cases and18draft checks match; actual MoE grouping reduces logical nodes720->240 without a floating-path change. Independent prefill variability must not be mistaken for candidate arithmetic drift. But shared-state success also cannot erase an earlier full-run failure: allocator/graph history differs, and full prefill/NLL/fault/native gates remain separate. Preserve raw changed host blobs only after the original failing operation completes to avoid perturbing the operation being diagnosed.


## Capture after the original read/hash boundary, and state which branch a metric exercises

[Nine first8K raw controls](experiments/1017-prefill-forensic-20261105/REPORT.md) and a [complete two-corpus conditional-capture repeat](experiments/1017-full-precision-forensic-20261105/REPORT.md) pass, but do not erase earlier uninstrumented failures. Keeping the original GPU read/hash order and dumping only after a mismatch avoids normal-path extra GPU reads; retaining host vectors and changing JSON/executable layout can still perturb allocator/timing histories. Record those limits.

The old NLL128 loop uses4-query chunks and cannot independently validate a selector active only at6queries. Report NLLdelta0 with its width and use the actual six-query full-vocabulary logits/state/rollback and dispatch evidence for the proposed mechanism. Matching repeated runs is observed evidence, not proof of universal baseline determinism or a root-cause fix.


## Raw width latency and native TG need separate matching controls

[MoE5+1 timing](experiments/1017-verify-moe5plus1-performance-20261105/REPORT.md) reduces raw six-query target latency20.26% at8K and15.84% at88K with exact outputs/state. Native gain depends on how often the same frozen policy reaches six queries. K5/p-min0.5 is not five proposals every cycle, and different prefixes have different aggregate acceptance. Use bracketing OFF/ON/OFF, same input IDs, full uncached prompts, exact output arrays and proposed/accepted counters, with all warmups retained. Compare within a prefix; neither raw latency ratios nor cross-prefix TG isolate total-engine improvement. Keep the255-interval versus256-token HTTP TG convention explicit. A short native pass still does not erase an earlier unresolved prefill/rollback failure or authorize default promotion.


## Separate a generic sample grid from shared-weight kernels

[Phase B float6](experiments/1017-verify-float6-20261106/REPORT.md) keeps the existing CUDA shared-weight selector capped at5 and changes only the graph's six-query sample representation.360 primitive evaluations and a paired full-model width/state screen pass exactly. Raw six-query latency improves only3.44%/3.13% at8K/88K, which is not a server TG result. Test launch grouping separately from a subsequent shared-weight selector extension; retain negative widths and fresh module identities after a full rebuild. Operator success and width restores do not replace rejected-window rollback, taps, failure recovery or native acceptance controls. When adapting a runner, resolve and assert its output root before execution; preserve a misdirected invocation and restore archived files from the committed evidence.


## A passing perturbed diagnostic does not clear a native mismatch

The [floating6 acceptance continuation](experiments/1017-verify-float6-acceptance-20261106/REPORT.md) passes full two-corpus state/rollback/fault/NLL6 and halves counted floating nodes, yet clean ON generation differs at the tenth8K output token. Fresh OFF repeats match. The ON trace then matches the OFF continuation: logging/binding/CPU allocation changes can hide the failure even without extra GPU reads or synchronization. Preserve the original failed manifest, classify subsequent controls and traces separately, and withhold native speed/default promotion until an uninstrumented reproducible comparison passes and the discrepancy is understood. Do not infer GPU sampling: these runs use CPU target sampling and GPU draft sampling.


The subsequent clean ON repeat after fdf8e354a also matches original OFF, including all 256 output tokens at8K and draft counters. Therefore the earlier diagnostic pass does not establish that its interposer specifically hides the failure: it also fails to reproduce without instrumentation. Preserve the clean failure and both kinds of passing repeats; do not claim a fix or attribute causation to logging from these observations alone.

## An empty or single-row fusion policy can discard the worst-case PP reserve

The [Qwen3.8 port](experiments/qwen38-port-20261106/REPORT.md) found a 7082-node reserve becoming 7081 for no-output PP and 7034 for decode. The first change fixing only empty rows accelerated the first cold PP but left later requests slow. The allocator then planned each growing cache from a small layout, adding about3.9 seconds of reserve-related waits per PP in the earlier audit. Kernel-level gains did not resolve this.

Separate execution selection from conservative allocation dependencies. Keeping dependencies for zero/single/many rows preserved 7082 nodes across PP/TG/PP/TG, with86 allocation calls and zero replans, without removing synchronization or changing the decode kernel policy. Always measure another request after decode, not only the initial PP. Preserve the failed intermediate variant and validate snapshot, rollback and producer/consumer lifetime.

## Rollback and hidden export need their own PP/TG reserve check

The [Qwen MTP follow-up](experiments/qwen38-mtp-20261106/REPORT.md) found a second topology mismatch after the plain port was fixed: conditional initial-state preservation made short rollback batches7449 nodes and PP7377. Removing only an output-copy fence improved cold PP but repeated PP remained slow. Empty initial-state views retain allocation dependencies without overwriting a real snapshot; together with an ordered producer-stream copy, the final graph stays7485 nodes with no repeated reserves. Include target/catch-up boundaries and at least one decode-to-prefill transition when testing a draft mode. Do not infer a large GDN bottleneck merely because MTP selects AR instead of MMA.

The real server follow-up also shows that higher acceptance alone does not establish useful speculation: shared MTP maximum1 accepts75.81% but TG59.07 is near the plain58.88-59.58 controls; maximum2 accepts66.96% and falls to56.67. Include prompt catch-up in latency: both MTP settings make a21K+256 request slower here. These settings/corpus are evidence, not a recommendation to disable MTP for every model. Preserve controller cleanup failures separately from completed inference responses; use SIGTERM for this fork's double-Ctrl+C shutdown behavior.

The [original MTP follow-up](experiments/qwen38-original-mtp-20261106/REPORT.md) demonstrates why a startup fix and a speed claim must be separated: one metadata integer in a private ordinary GGUF copy permits an unpatched original run, while source files/model weights stay unchanged. Compare the exact same draft file before attributing a runtime gap. Our adaptive draft policy is an independent mechanism: disabling it improved this series' median, but one repeat regressed strongly, and fixed-policy full/shared comparisons do not prove their tensors differ. Check the constructed context's stored ctx_other before alleging shared-KV behavior; original Qwen clears it, so the preliminary suspected driver defect was inactive.


## Reuse graph identity before changing tensor-input ownership

The [Qwen Strata-inspired experiment](experiments/qwen38-strata-port-20261107/REPORT.md) reduces CPU hashing by reusing the existing canonical CUDA graph key for an immutable nonzero UID. UID0, eviction and changed topology retain the full hash. Most measured hash time overlapped GPU execution: report its intersection with no-kernel intervals instead of subtracting total host time from token latency. Decode branch overlap also requires allocator dependencies at empty/prompt row widths; a decode-only metadata-pack prototype passed exactness but made later PP slow. Preserve graph-history controls and another full PP after decode.

An initial GDN fusion40-head guard never ran because this model has48 value heads. Correct the guard/output-node indexing, confirm the actual demangled kernel name (boolean template parameters may appear as `(bool)1`), then rerun gates. The active exact variant still gave no native speed gain and was removed. Thread20/12/4/1/20 showed a returning20-thread control as fast as1/4, so the first20-thread result did not establish a threading bottleneck. Async inputs and packing were also removed with patches/evidence retained. Repeated PP can increase with CUDA graph-cache warmup even when prompt_n is unchanged and cache_n is zero; retain request ordinals and first-request timing. GPU placement can change sampled IDs under the same seed, so a four-GPU TG screen with divergent output is not an exact-output speed comparison.


## Routed-expert rank locality can matter without changing reductions

The [GLM rank-first grid experiment](experiments/glm53-moe-real-inputs-20261110/REPORT.md) captures actual expert IDs/activations before selecting a grid change. Repeated experts mostly occupy different ranks across tokens; exchanging row/rank launch axes preserves per-row math and improves narrow Ada N5 fused-up/gate windows and actual DFlash TG. Capacity/logical footprint observations do not measure cache traffic or guarantee CUDA block order. Preserve input strides, GLU clamp and real weight types; use unique first-node graph keys so policy switches do not silently recapture. Retain failed shared-anchor timing evidence and reject architecture-specific timing claims when unchanged controls move. An Ada-only refinement plus sham old-kernel arm separates the full-window candidate from drift. Match server IDs/counters and both OFF brackets; distinguish external-module speed evidence from subsequent native source-port exactness. A passing narrow cohort does not resolve historical restart variability, establish N1/PP improvement, or authorize default promotion.


## Close the external-module gap with a no-preload native bracket

The [GLM native rank-first follow-up](experiments/glm53-moe-native-server-20261110/REPORT.md) uses the source-integrated backend for unset/ON/unset, with identical loaded module hashes and no math or metadata preload. This retains the earlier bounded workload's output IDs/counters and gives +2.50..2.53% TG versus both controls while OFF medians drift only0.028%. Keep the native measurement separate from external-module/operator percentages and leave source/default promotion explicit. Byte-identical original-object relink plus a single changed object makes provenance reviewable; native operator activation is prior diagnostic evidence, not new full-server kernel/tensor introspection. A clean bracket closes that specific speed gap, not historical cross-start variability, per-window state/lattice coverage, ordinary decoding or other-model quality.
