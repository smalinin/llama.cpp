# Каталог механизмов и границ применимости

Состояние на runtime `b3377774b` + проверенный block-local delta/defaultON (pooling OFF), 2026-11-03. Shipped/default означает наличие dispatch в этой ветке с guards. Нужные model types, размеры, backend и свободная память всё равно проверяются. Исторические проценты относятся только к перечисленным опытам.

| ID | Механизм и статус | Что можно переиспользовать | Что установить для новой модели | Локальное доказательство |
| --- | --- | --- | --- | --- |
| O1 | CPU MXFP8 LUT, shipped default | Табличный decode с сохранением scale/exception semantics | Реальный тип таблицы MXFP8, CPU GET_ROWS, exceptional blocks; Q5_K/Q8_0 этот LUT не ускоряет | [E02](evidence/README.md#e02) |
| O2 | Routed Q3_K MMQ tile32, shipped default в подходящем dispatch | Выбор тайла по фактической форме и GPU, whole-K reduction | Число экспертов, rows/expert, chunk, types, Ada/Ampere; Q2/Q4 и другие GPU требуют своих gates | [E03](evidence/README.md#e03) |
| O3 | GPU-only overrides + pipeline input ownership, shipped в ограниченном DS41 пути | Eligibility по actual buffers, несколько input copies, reuse после completion event, metadata/real parity | Host spill, cache/staging, consumers, split, offsets и scheduler lifetimes; разрешённый override не доказывает overlap | [Текущий код](../../src/llama-context.cpp), [режимы](../../readme_dsv41.md) |
| O4 | Encoded Q8 packing, compact/counts MMA, dual raw/compressed sources; shipped conditional | Бounded unpack/gather вместо обработки всей history, явные counts/masks, shared KV reads | Q8 layout и rounding, head geometry, sinks, Top-K, padding, masks, GPU и reserve; F16 reference и same-Q8 reference различать | [E04](evidence/README.md#e04), [E13](evidence/README.md#e13), [README](../../readme_dsv41.md) |
| O5 | Batched target verification и sample batching, shipped conditional | Сократить launches для нескольких independent queries, сохранить causal/reduction semantics | Draft block width, all-logits outputs, rollback capacity, layout, kernels; plain GEMM может менять rounding | [README](../../readme_dsv41.md), [source](../../common/speculative.cpp) |
| O6 | Producer-local DSpark taps, shipped default; ASYNC отдельно OFF | Хранить данные у реального producer, собирать нужный chunk у consumer | Tap owners после scheduler, encoder device, copy costs, peer fallback, events и metadata budget; новый consumer требует собственного контракта | [E05](evidence/README.md#e05) |
| O7 | M2 source-KV-only prefill + DSpark tail-seeding, shipped default | Dependency-cone pruning и явный suffix contract; продвигать metadata только внутри закрытого seed с полным overwrite кольца | Точная KV/index topology и все consumers; DS41 source20/W128/taps37-39/C768 не универсальны; U512/P1/B<=8192/allGPU guards | [E01](evidence/README.md#e01), [паспорт](models/deepseek41-1017.md) |
| O8 | Mixed expert cache, byte budget/profile/spare; cache включается отдельно | Проверка publication/lifetime, actual host projection bytes, popularity profile, bounded replacement | Geometry gate/up/down, host/GPU routing, cache owner, полезные/физические slots, prefix/decode route distributions | [README](../../readme_dsv41.md), [source](../../src/llama-moecache.cpp) |
| O9 | Startup CPU/staging calibration и auto selection, opt-in auto | Измерить конкретные lanes и переиспользовать cost cache с fingerprint | Actual CPU/PCIe/GPU/types/shapes/workspace, stale invalidation, overhead calibration; текущий host-Q3 staging scope не универсален | [E06](evidence/README.md#e06) |
| O10 | HC root/full-reshape destination sharing, guarded default ON по запросу пользователя | Одна protected CUDA copy, allocation-only view и корректный n_views/lifetime, same-split ownership | Только read-only HC_POST/F32/direct/contiguous/offset0 и RMS_NORM, n_copies>1/CUDA; unknown consumers/callback/иные views fallback | [Дизайн](designs/1017-hc-copy-sharing.md), [проверки/promotion](experiments/1017-hc-copy-sharing-20261103/REPORT.md) |
| O11 | Candidate pre-pooling у score producer, experimental opt-in OFF | Выполнить CONT/padding/reduction до cross-GPU transfer; сохранить selector placement и численные ops | Actual producer/cache device, source topology, full chain/next owner, supported ops, mask semantics, IO slots, precise layout proof | [Дизайн](designs/1017-indexer-placement.md), [gates](experiments/1017-candidate-pool-local-20261103/REPORT.md) |
| O12 | Compact block keep transport/local expansion, guarded default ON по запросу пользователя | Передавать меньший промежуточный tensor и повторять byte-preserving expansion у consumer, без новых streams/events | Actual producer/query owners, source domain, graph reuse key, padded strides, protected per-slot copies, full-range pointer proof; block TopK не равен final row TopK | [Дизайн](designs/1017-candidate-block-local.md), [gates](experiments/1017-candidate-block-local-20261103/REPORT.md) |
| O13 | Independent CPU GET_ROWS overlap, guarded default ON by user request | Start an isolated lookup before a preceding GPU fence; reuse protected host staging and input slots | Prepared async inputs, distinct backend buffer types, immutable host weights/owned row IDs, no host compute read by CUDA, current and previous graph isolation; other models/cache/media/parallel require gates | [Design](designs/1017-cpu-lookup-overlap.md), [new-host checks](experiments/1017-cpu-lookup-overlap-20261105/REPORT.md) |

## Результаты, которые стоит помнить

- O1: CPU decoder быстрее примерно в4 раза в своём screen, но whole-model PP8K вырос лишь на +0.49..0.82%; PP512 +4.24%. Доля операции ограничивает общий gain.
- O2: Q3/BF16/F16-KV native B8192/U512 PP512 +28.09%, PP8K +24.39%; TG практически прежний. Это tile0->32 на прежнем build, а не дополнительный gain к последнему M2.
- O6: LOCAL PP8K 787.41 против CPU772.35 и central761.65 в одной fixed-order серии. Поздний98K LOCAL741.51 против CPU730.44; это небольшой transport эффект, а не ускорение всего draft на проценты transfer microbenchmark.
- O7: на Q3/Q8KV/BF16 K3 PP8K 788.38->1045.25 (+32.58%), PP98K741.61->1065.66 (+43.70%). TG48.57->48.48/46.22->45.92; exact response IDs/draft counters. Подробнее [E01](evidence/README.md#e01).

## Кандидаты, не прошедшие promotion

| ID | Кандидат | Результат и следующий разумный шаг |
| --- | --- | --- |
| N1 | Последовательный per-expert FP16/GEMM | Aggregate operator throughput при threshold64 Ada -18.38%, Ampere -2.18%; offline classification даже не включена в cost. В runtime не перенесён. Новый grouped/GPU-driven вариант оценивать отдельно. [E07](evidence/README.md#e07) |
| N2 | FFN-only CUDA graph regions | Exact correctness и реальные replays; PP512 +0.16%, Wiki8K -0.02%, code8K +0.03%. Полезный PP gain не установлен; не интегрирован. [E08](evidence/README.md#e08) |
| N3 | Центральный GPU bridge вместо CPU-overlapped reference | Несколько screens дали около -1.5% PP. Transport route и fences оказались важнее packing kernels; LOCAL стал отдельным кандидатом. [E09](evidence/README.md#e09) |
| N4 | ASYNC ingress поверх central bridge | PP8K -0.43%, TG -1.88% в одном порядке; ASYNC остаётся OFF. LOCAL+ASYNC отдельно не закрыт. [E10](evidence/README.md#e10) |
| N5 | Engram lookup ownership/prologue как самостоятельный PP фикс | Correctness прошёл, PP8K +0.54% text/+0.06% code; заметный gain не установлен. Новый вариант должен показать полезное overlap, а не перенести ожидание. [E11](evidence/README.md#e11) |

Отдельное сравнение F16/Q8 сохранило memory advantage Q8, но TG был ниже при том же placement; [E14](evidence/README.md#e14). Same-Q8 kernel gain и F16->Q8 tradeoff - разные эксперименты.

## Открыто

Q2/Q4 MMQ, другой model graph, multimodal/parallel>1, actual OOM/cancellation recovery и свободный joint auto-fit262K не закрываются перечисленными DS41 gates. Historical evidence различает controlled compute failure и реальные ошибки памяти. Для другого контекста/pipeline placement повторить reserve/actual memory проверки.

## Residual profile после M2: что проверять дальше

[Новый отчёт](experiments/1017-m2-residual-profile-20261103/REPORT.md): 19 запросов, exact IDs/counters, floor512, без runtime changes. Routed Q3/Q4 занимает 54-57% kernel sum и остаётся основной GPU работой. Getter12.148s на98K включает target wait; draft seed kernels19ms. Узкий input-copy wait перед internal alloc timer - лишь3.371ms в diagnostic request; расширение copies4->8 не получило обоснования.

Следующий read-only аудит: logical CUDA0->1 l_last-6 и reshaped имеют по8.771GB/count232 в общей diagnostic серии; нужно проверить root/data/offsets/strides/consumers и actual DMA перед dedup. Large idx_score-20/candidate_mask-20 edges тоже требуют анализа. Счётчики scheduler и физические CUPTI copy bytes - разные величины. Это открытые кандидаты, не реализованные оптимизации.

## M2 larger context и HC alias audit, 2026-11-03

[Memory/transfer gate](experiments/1017-m2-large-context-20261103/REPORT.md): fresh default/OFF no_alloc budgets exact; actualctx176128/172032+16 прошёл с minfree601.25MiB,15360GPU bundles и exit0. Output IDs exact к historical pre-M2 LOCAL continuation;8K256 exact к M2 benchmark. Это fixed placement, не solver proof для262K.

Следующий узкий transfer кандидат подтверждён на уровне API+metadata:32 root/zero-offset contiguous reshape pairs,64 peer-copy requests в разные destinations,1.342GB потенциально устранимых повторных API bytes. Allocator address reuse исключает трактовку всех pointer matches как HC traffic. Physical DMA/cost и safe destination view sharing ещё требуют проверки. Код/defaults не изменены.

[Bounded HC sharing design](designs/1017-hc-copy-sharing.md), реализован и проверен; default ON по запросу пользователя: protected pipeline slots, same-split direct/full/F32 reshape of HC_POST, canonical root transfer unchanged, allocation-only destination view before consumer. Root/view aliases нельзя связывать одной подменой data pointer: allocator должен учесть n_views и source lifetime. [Historical cost screen](experiments/1017-hc-copy-design-20261103/REPORT.md) отдельно от будущего native A/B.

HC sharing native: PP512+4.06%,8K+4.26%,98K+3.77% к среднему OFF-pre/post; TG -0.50..+0.37% в screen,98Ksingleton. Combined reserve -640MiB, model/KV/routes прежние. Full target/draft/logits/taps/state/rollback exact,NLL delta0. Selector promotion отдельно проверил unset==1; данные measured phase не переписаны.


## HC default ON:176128 и следующий indexer шаг

[Actual176128/defaultHC ON](experiments/1017-hc-long-indexer-20261103/REPORT.md) закрыл прежний no_alloc-only пробел:172032+16 и повтор8K прошли,761.25MiB sampled floor,placement/IDs exact,exit0. [Новый indexer design](designs/1017-indexer-placement.md) на момент аудита был proposed/defaultOFF; реализация O11 и последующий O12 описаны ниже.72 source-score copies по8.782GB суммарно и72 dense-mask copies того же объёма; pooled scores8x меньше. Это размер payload, не end-to-end gain. На8K candidate branch отсутствует. Следующий эксперимент должен сохранять CONT/pad/pool численную семантику и independently проверить native PP/TG/memory/state/rollback.


## Candidate pool: результат bounded prototype

[O11](experiments/1017-candidate-pool-local-20261103/REPORT.md): PP98K1106.39->1122.53 (+1.46%) к bracketing OFF, TG46.23->45.73 (-1.07%,singleton). На8K PP-0.28%/TG-0.29%.48 raw result groups exact,actual176128 min761.25MiB. Source/routing masks прежние; no_alloc -1073MiB combined compute при176K освобождает преимущественно CUDA3 и не меняет bottleneck CUDA4/global floor. Не выводить8xPP из8xменьшего payload. DefaultOFF, startup-only operational use; no broad model/KV/media support claim. Следующий отдельный кандидат: block keep transport/local expansion.


### O12: block keep transport/local expansion

[Отчёт](experiments/1017-candidate-block-local-20261103/REPORT.md): pooling1 фиксирован, PP98K+2.93% к pool1/block0, TG gain не установлен (-0.62%singleton). Wide dense-mask CUDA3->4 zero;134compact requests3.338GB вместо26.700GB, protectedslots0..3/actualREPEAT/fullptr proof.24 precision groups/deltaNLL0/4seedfaults,operator98/fixed32capture/externalfailure-retry exact. Combined compute reserve -520.52MiB@176128;actual172032+16/repeat8K min1143.25MiB,15360GPUbundles/routes/IDs exact,exit0. LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL unset/1ON по запросу пользователя,0OFF;фиксируется до загрузки. Pool0metadata controls не заменяют pool0native gate. BLOCK_LOCAL promoted defaultON,POOL_LOCAL остаётсяOFF;7freshno_allocdefaultchecks passed.


## Текущие candidate defaults: native gate закрыт

[Отчёт](experiments/1017-pool0-default-20261103/REPORT.md): runtime `my_maya 0e6f3cec3` и defaults не менялись. Pooling не задан (OFF); в default arm BLOCK_LOCAL также не задан (ON). На двух 98K text/code корпусах 24 группы logits/taps/state/rollback совпали побитово, delta NLL=0; четыре seed failures/retry прошли. Native OFF/default/OFF: 27 output arrays и draft counters совпали. PP98K 1107.75 -> 1138.16 t/s (+2.75%); TG 46.21 -> 45.82 (-0.84%, по одному измерению на arm, ускорение TG не установлено).

Actual ctx=163840 / prefix=159744 / repeat 8K прошёл: минимум 679.25 MiB свободной VRAM, все 15360 экспертных bundles на GPU, placement совпал, exit 0. Context=176128 не допущен расчётом при pooling OFF; прежний actual 176128 с pooling=1 остаётся отдельной серией. Это fixed Q3/MXFP8 Engram/Q8 KV/BF16 DSpark K3/P1 конфигурация, не универсальный предел контекста.

Ошибка post-test сборки сводки (`queue.py` перекрыл stdlib) вызвала повтор metadata. Исходные precision/native/payload manifests сохранены побайтно; память и routes metadata сверены с исходными admission inputs. Incomplete manifest, старые hashes и recovery diagnostics сохранены в отчёте. Runner переименован и защищён от выполнения при импорте.

Далее: matched Maya comparison, свежий residual profile, затем Q2/Q4 MMQ. Media, parallel>1 и auto-fit 262K остаются отдельными проверками. Нового commit/deploy нет.


## Новая платформа, 2026-11-05

[Новая независимая серия](experiments/1017-maya-newhost-20261105/REPORT.md): Threadripper PRO 3945WX, 192 GB RAM, новый PCI/UUID-порядок GPU, CPU threads12. Все прежние timings и memory/cost estimates относятся к другой машине. Выполнено сравнение PP/TG без sidecar и с тем же sidecar на 512/8K/98K, с actual memory floor и fresh calibration. Результаты скорости, CPU, acceptance и repeat limitations приведены в отчёте. Далее свежий профиль prefill/verification; runtime/defaults не менялись.


## Новая платформа: подробная телеметрия Maya и нашего runtime

[Отчёт](experiments/1017-maya-telemetry-20261105/REPORT.md): на записанном prefill8K условное число вычисляющих GPU Maya3.33, наше1.32;98K3.93/1.36. Polling/transport исключены, это не SM occupancy. Maya исполняет все40слоёв, наш pipeline enabled, но фактическое перекрытие слабее. TG profile указывает на target fragmentation и guard: anchor+5draft даёт6queries, KV-wide допускает6, обычный float/MoE batch guard5. Это кандидаты следующих native опытов, не реализованный фикс.

Профилировщик снижает наш TG примерно на27-29%, есть CUPTI GraphId diagnostics; CPU sampling/OSRT недоступны. Maya98K восстановлен после STOP/controller timeout без post-control; original manifest остаётся incomplete. Полный8K и оба наших захвата сохранены, runtime/defaults прежние. Нельзя выдавать recorded idle/API time за полную native critical-path decomposition. Следующий приоритет: bounded chunk/input/copy/feature lifetime для PP, counters-only sizes1..6 для verification.


## New-host residual prefill: queue depth and copy slots rejected by screen

[Evidence](experiments/1017-prefill-residual-20261105/REPORT.md), runtime f796e544f with guarded CPU lookup ON. Existing CUDA queue4x and private `GGML_SCHED_MAX_COPIES=8` do not provide useful native PP8K gain in the fixed Q3/Q8 KV/DSparkK5 layout. They remain unadopted; production copies4 and launch-queue default are unchanged. Host/API timing is diagnostic, not a kernel speed claim. Shared gate/up6/29 co-location removed24 backward transitions but PP changed-0.65%, so it is not adopted. A new scheduler dependency pattern still requires reviewed ownership/lifetime design and raw precision/state/rollback gates.


## Deferred remote cache writes: existing decode path, proposed prefill extension

LLAMA_DSV41_DEFER_CACHE_UPDATES is default ON in source for guarded DeepSeek41 decode/verification with single sequence/full GPU attention/layer split; it is inactive on512-token prefill. The [owner audit](experiments/1017-kv-owner-audit-20261105/REPORT.md) identifies eager LID/CSA fanout, actual source transform on a remote device, and protected slot reuse. The user approved the separate [prefill design](designs/1017-prefill-cache-deferral.md). The [opt-in implementation](experiments/1017-prefill-cache-deferral-20261105/REPORT.md) passes paired exactness/state/rollback/NLL and actual owner/KV-only-flush checks; native OFF/ON/OFF completed at identical ctx92160: PP8K+4.45%, PP88K+7.40%, PP512-4.74%, sampled floor535.25MiB. No scheduler fence removal or default promotion. Initial Q8_0 text-only scope and memory cost are recorded in that report.


## Approved prefill KV deferral: exact but memory-costly opt-in

[Completed evidence](experiments/1017-prefill-cache-deferral-20261105/REPORT.md): same private module with flags0/1/0, ctx92160/longprefix90112, new-host fixed Q3/MXFP8 Engram/Q8 KV/DSparkK5/full/pmin0.5/U512/P1/15360GPU experts. PP512724.48->690.17t/s (-4.74%), PP8K1412.14->1474.92 (+4.45%), PP88K1449.66->1556.94 (+7.40%); TG gain not established. All27native output arrays/draft counters,12paired precision groups/two88K corpora/NLLdelta0/four failure-retries exact. Actual source20 producerCUDA4->CUDA2, pending KV-only flush and419raw/CSA/LID writes per arm verified. CUDA5 joint reserve+795.02MiB, native minimum535.25MiB; higherctx100352..94208 failed conservative admission. Flag LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1 opts in; unset/0 OFF. Code available for review, production modules/presets unchanged; no new commit/default promotion. Do not infer broad auto-fit262K/F16/media/parallel/CPU-cache support. Workspace and short-prompt regression are separate follow-up candidates; six-query verification remains independent.


## New-host verification width screen: six-query fragmentation measured

[Completed screen](experiments/1017-verify-width-screen-20261105/REPORT.md), after runtime commit3854c3785: fixed prefix/tokens/state/n_rs_seq5, raw target decode+sync medians at8K:5queries43.03ms,6queries68.55ms (+59.30%);88K46.02->68.17ms (+48.12%). Separate lightweight counters show MoE120->720 and float275->950 matmuls, backend graphs44->114 /43->113. All156per-arm logits/splits match native/counter;24sampled state checkpoints per arm and base restores exact. Actual floor1391.125MiB, exits0, source/modules unchanged. Cold smaller-width graph instantiation rows are retained; measured width6has0instantiations and all llama graphs reuse. This is not total server TG/acceptance or a new implementation/default. [Next bounded plan](designs/1017-six-query-moe-float.md): isolate grouped MoE6, then floating sample-grid6; shared-weight MMVF6 is a later separate kernel candidate.


## Six-query MoE5+1: scoped correctness evidence, default OFF

`LLAMA_DSV41_VERIFY_MOE_WIDE=1` is the committed opt-in for the bounded DeepSeek41 six-query/K5 path, with Q8 CSA/LID, text/P1, full CUDA expert placement/pipeline and separate contiguous Q3_K/Q4_K experts. Direct6 was rejected: Ampere Q3_K MUL_MAT_ID switches away from MMVQ above5, unlike Ada. The5+1 path leaves the CUDA selector and generic floating cap unchanged.

[Shared-prefix evidence](experiments/1017-verify-shared-prefix-20261105/REPORT.md) passes117cases/18draft checks and observes MoE720->240/backend graph calls113->57, floating950 unchanged. [Nine raw OFF first8K captures](experiments/1017-prefill-forensic-20261105/REPORT.md) and a [complete two-corpus conditional repeat](experiments/1017-full-precision-forensic-20261105/REPORT.md) also pass, including restore/rollback and two failure/retries; actual floor1387.125MiB. NLLdelta0 uses the original4-query loop, while6query exactness has separate full-logit/state evidence. Earlier uninstrumented failures are unresolved. No default promotion, broad hardware/model coverage, native speed or production deployment is implied. Next is controlled cost/native measurement; mismatched controls must be retained and excluded from equivalent speed comparisons.


## Six-query MoE5+1: controlled native speed PASS, default OFF

[Completed performance evidence](experiments/1017-verify-moe5plus1-performance-20261105/REPORT.md), after afc37abd8: same private9c440a18 module/flags0-1-0, prefill deferral0, new-host Q3/Q8/K5/full/p-min0.5/ctx92160/U512/P1/15360GPU bundles. Raw six-query target medians8K68.571->54.680ms (-20.26% latency),88K68.162->57.361ms (-15.84%); widths1..5 stable. Native 512tokens TG41.55->42.86t/s (+3.15%); 8192tokens TG45.64->48.60t/s (+6.49%); 90112tokens TG60.90->71.51t/s (+17.42%). All27 output arrays/draft counters,156raw observations and24state checkpoints per arm match; warmups excluded from speed. Native minimum1329.250MiB, exits0; no source/default/production-module changes. This short screen does not resolve the old uninstrumented failures or certify other modes. Next: separate floating sample-grid6, then shared-weight MMVF6. Flag remains opt-in/defaultOFF.


## MoE5+1 default ON by user request

[Promotion checks](experiments/1017-verify-moe5plus1-default-20261105/REPORT.md): unset/1 now enables `LLAMA_DSV41_VERIFY_MOE_WIDE`; 0/off disables. Existing six-query/P1/text/Q8/full-CUDA/Q3_K-Q4_K/pipeline guards and fallback stay unchanged. Four fresh no_alloc variants verify reserved allocations/routes against the previously measured explicit ON/OFF candidate, all15360 GPU bundles. This default change does not fix or erase the historical uninstrumented failures, extend mode coverage or enable Phase B floating6. Prior controlled performance gains are +3.15%/+6.49%/+17.42% TG at512/8K/88K in the recorded K5/p-min0.5 setup.


## Floating sample-grid6: first gates PASS, default OFF

[Phase B evidence](experiments/1017-verify-float6-20261106/REPORT.md), after db4be119e: separate `LLAMA_DSV41_VERIFY_FLOAT_WIDE=1/on` opt-in; MoE5+1 explicit1 in both full-model arms, prefill deferral0. Reuse existing sample view/GEMV, no CUDA or global five-query predicate changes.360 representative F32/F16/BF16 operator evaluations on Ada/Ampere are bitwise exact. Fresh paired width1..6 on rebuilt modules matches all156outputs/24sampled state checkpoints per arm; six-query target medians8K54.654->52.772ms (-3.44%),88K57.357->55.561ms (-3.13%), actual minimum1395.125MiB, exits0. This is raw target latency, not native TG. Dispatch counters, complete two-corpus lifecycle/rollback/fault, MoE0-only and native OFF/ON/OFF remain next; no default promotion or new commit. Initial runner-path error is preserved and its overwritten historical files restored from HEAD. Earlier uninstrumented failures remain unresolved.


## Floating6 acceptance continuation, 2026-11-06

After c6be3d990, the [combined floating6 lifecycle/dispatch continuation](experiments/1017-verify-float6-acceptance-20261106/REPORT.md) passes14groups per arm on two88K corpora: logits/taps/state/restores/append/rollback/fault recovery, NLL128/4 and NLL126/6 delta0. All56six-query counter pairs show floating MUL_MAT950->475, unchanged MoE240/quant331/backend calls57. These are logical counts, not GPU kernel timings. Clean native acceptance then FAILS on the ON8K warmup at generated index9 (OFF469, ON27344); eighteen OFF responses across two fresh processes match. Subsequent diagnostic ON/OFF matches all eight responses,1365 decode events and22 captured logit rows per arm; it does not reproduce or supersede the clean failure. Keep FLOAT_WIDE default OFF; no accepted native gain, default promotion or production deployment. Historical uninstrumented failures remain unresolved; MoE0-only/other policies remain outside this cohort.


After documentation commit fdf8e354a, a [fresh clean ON reproduction](experiments/1017-verify-float6-acceptance-20261106/REPORT.md#clean-on-reproduction-after-fdf8e354a) repeats the original512-to8K history without any interposer: all four 256-output arrays and draft counters match original OFF; the8K tenth token is469, not the failed ON27344. The mismatch does not reproduce in this ordinary repeat either. Its cause remains unknown; preserve original failed acceptance, keep FLOAT_WIDE OFF and do not infer a native speed gain.

## Shared CUDA MMQ on GLM-5.3-Flash (2026-11-06)

[Experiment](experiments/glm53-mmq-20261106/REPORT.md), [passport](models/glm53-flash.md). Shipped guarded default: upstream linearized MoE tiles and gate/up/GLU MMQ fusion, with matching MMF index descriptors. The existing Q3_K tile control is retained. This shared FFN/MoE path is selected by type, geometry, operation semantics and device support; unsupported cases keep their fallback. Native PP50K improved 18.1476% on GLM UD-Q5_K_XL at B2048/U1024/F16-KV/P1 on Ada/Ampere. Matrix/fusion CPU-reference checks and GLM/Qwen4Exp state checks passed; other real model speedups and Blackwell/HIP are not inferred from this result.
