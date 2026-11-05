# Паспорт: DeepSeek-V4.1-Flash-1017 в my_maya

Дата снимка: 2026-11-03. Runtime `b3377774b` + проверенный candidate-block-local delta/defaultON. Это паспорт проверенной Q3 конфигурации, а не спецификация всех квантов/размещений DeepSeek. Границы взяты из current source и [E01](../evidence/README.md#e01).

## Архитектура и данные

| Свойство | Проверенный вариант |
| --- | --- |
| Target | deepseek41, 40 blocks, hidden5120, 384 routed experts/block, 6 picks, expert width2304 |
| Веса | Q3_K-AntirezLike mixed recipe, 10 shards; реальные types проверять по tensors |
| Engram | Две таблицы MXFP8 в shards8/9; lazy CPU/mmap, не полный upload таблиц в VRAM |
| Top-K indexer | Явный override2048 |
| KV в последнем M2 speed test | Q8_0 K/V; исходный CLI default F16 этим не менялся |
| KV topology M2 | Source20, upper suffix20..39 с shared persistent KV/index; W128 |
| DSpark | BF16, arch dflash, trained semantics DS41, taps37/38/39, Markov/confidence heads |
| Draft seed ring | Effective SWA129, physical capacity768; 3 stages, encoder input3x5120 |
| Execution | CUDA layer split, P1/text, B8192/U512, no media/callback/adapters/cache/staging |
| GPUs | 6, mixed Ada/Ampere; FASTEST_FIRST; logicalCUDA5 RTX3090 24GiB, остальные48GiB в этом layout |
| Target/draft placement | Frozen all15360 routed bundles GPU; draftCUDA5/CUDA0, GPU overrides из [manifest](../evidence/reproduction/dspark-prefill/bench-results.json) |
| Context speed fixture | 100352; reserve floor512MiB, measured server minfree1445.25MiB |

## M2: что именно сохраняется

Источник20 публикует всю необходимую KV/index history. Для target-only полного состояния получается bound3054, округлённый full tail3072 при U512. С DSpark используется другой cut: начало доступных features `floor((N-768)/512)*512`, и `cut <= first_valid - 18*127`, округление вниз512; также действует прежний target raw/output bound.

| Logical N | Cut с DSpark | Полный target tail | First valid tap row | Реальный injection tail |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 4608 | 3584 | 7168 | 1024 |
| 7676 | 4096 | 3580 | 6656 | 1020 |
| 7681 | 4096 | 3585 | 6656 | 1025 |

Это относительные rows текущего batch; абсолютные позиции/seq0 проверяются в seed. Нельзя заменить physical768 на live129 или всегда брать последние два ubatch. До корректного хвоста draft продвигает только metadata в закрытой транзакции, затем обычные512/partial projections перезаписывают всё кольцо. Ошибка seed очищает оба контекста и требует full prefix reload.

`LLAMA_DSV41_PREFILL_SKIP` default ON после решения пользователя: unset/1 разрешает guarded path,0 отключает. При B2048 skip не активируется. Иные consumers, topology, host experts/cache, media, parallel, eval/abort callbacks, LoRA/cvec и async feature input сохраняют fallback.

## Проверки и числа

CPU108 ring fixtures, short native24, long98K30 групп/36 rollback workflows/2 controlled failures, transport24 и final lifetime12. В пределах fixture target/draft logits/seq/global states, real taps,K3/K5/confidence, restore и replay exact. NLL128 продолжения отдельно text/code delta0; это не PPL всех prompt tokens. Последний final packed-range guard проверен отдельной сборкой после timed A/B; default promotion позднее меняет selector при unset.

С BF16K3/full/p-min0.5/backend sampling ON, default LOCAL, ASYNC OFF, warm+3 x256 greedy:

| Prefix | OFF PP | ON PP | OFF TG | ON TG |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 788.38 | 1045.25 | 48.57 | 48.48 |
| 98304 | 741.61 | 1065.66 | 46.22 | 45.92 |

Один OFF->ON порядок и один speed corpus. Small TG deltas не интерпретируются как доказанная регрессия/ускорение. Не сравнивать напрямую со старым Maya903.18/52.91: context/KV/placement/K/output limits и timer boundaries различались. Свежий matched Maya test остаётся открытым.

Полный reproduction bundle и неизменяемые measured/final patches находятся в [evidence](../evidence/README.md#e01). Старые helper scripts требуют remapping recorded paths и сборки с соответствующими internal headers. Прежний pre-M2 actualctx176128 дополнен свежим [M2176128 gate](../experiments/1017-m2-large-context-20261103/REPORT.md):172032+16, minfree601.25MiB, placement exact, exit0. Это fixed-layout memory/lifecycle проверка; свободный auto-fit262K не закрыт.

## Следующий профиль после M2

[Residual profile](../experiments/1017-m2-residual-profile-20261103/REPORT.md): native8K/98K controls1050.63/1069.16PP, по одному измеренному control, не новая A/B серия. NSYS сохранил IDs/draft counters и M2. Routed Q3/Q4: 54-57% kernel sum; CPU token/Engram2.027s на98K; seed kernels19ms. Следующая гипотеза связана с cross-GPU tensor traffic и возможными дублирующими view copies; whole-context176128 проверен последующим этапом, ссылка выше.

## Аудит HC view transfers

[Actual memory/alias report](../experiments/1017-m2-large-context-20261103/REPORT.md): на одном8K+16 request16 wide frames,32 root/reshape alias pairs. CUDA0->1 l_last-6 и CUDA1->2 l_last-14 получают по две peer-copy в разные destination buffers с общим src/device/bytes/stream. Всего1.342GB повторных API bytes; matched destination consumers проверены. Это не CUPTI physical DMA и не измеренный PP gain. Этот кандидат реализован и проверен следующим этапом, ссылка ниже.

## HC sharing: текущий default ON

[Полный отчёт и promotion](../experiments/1017-hc-copy-sharing-20261103/REPORT.md). В frozen Q3/Q8KV/BF16K3/ctx100352/B8192/U512/P1 sharing дал PP+3.8..4.3% и combined reserve-640MiB.27 output arrays/draft counters exact;36target/draft groups,36rollbacks,2failure/retry,NLL delta0 отдельно text/code. TG выигрыш не установлен;98Kdelta-0.50% одиночная точка. По запросу пользователя unset/1 разрешает GGML_SCHED_COPY_RESHAPE,0 отключает; guards и fallback сохранены. HC delta включён вc3c91dda8; candidate-pool включён вb3377774b; block-local/defaultON описан ниже. Свежий [defaultON176128 gate](../experiments/1017-hc-long-indexer-20261103/REPORT.md):172032+16, затем8K16, sampled minimum761.25MiB>=512,actual15360GPU bundles/per-tensor placement exact,IDs exact к controls,exit0. DefaultOFF M2176128/601.25MiB остаётся отдельным историческим gate.


## Indexer placement: следующий bounded кандидат

[Audit](../experiments/1017-hc-long-indexer-20261103/REPORT.md) на current defaultHC ON подтвердил source20 ratio1, candidate block8/top2048blocks и independently final2048rows.72 candidate frames на98K: fullscore CUDA2->3, CONT/pool/selector CUDA3, densemask CUDA3->4 для queries32/36.107 из193 target ubatches KV-only,8K без candidate branch. Proposed [design](../designs/1017-indexer-placement.md) переносит CONT/padding/pool к score producer с сохранением ops/numerics. На момент аудита prototype ещё не был реализован; последующие O11/O12 приведены ниже. Full critical-path cost не заявляется из incomplete Nsight graph trace.


## Candidate pool: реализован, opt-in OFF

[Полный отчёт](../experiments/1017-candidate-pool-local-20261103/REPORT.md): source20/r1/block8/top2048blocks/final2048positions сохранены. CONT/poolCUDA2,selector/maskCUDA3,wide full-score copy2->3 исчезла;134 block-score requests и134dense-mask requests3->4 подтверждены exact pointers/protected clones/consumers. Query3/4 fallback прежний.48 raw precision groups,deltaNLL0,text/code; operator80boundary cases/fixed32capture exact. Final native98KPP+1.46%,8KPP-0.28%,TGgainнет/98K-1.07%singleton. No_alloc reserve -593MiB@100K/-1073MiB@176K,actual176128min761.25MiB,exit0. LLAMA_DSV41_CANDIDATE_POOL_LOCAL=1 задаётся перед загрузкой; unset/0 OFF. Mutable-fixture warning/ошибки стенда и final guard phase identities явно разделены в report.


## Candidate block local: реализован, guarded default ON

[Отчёт](../experiments/1017-candidate-block-local-20261103/REPORT.md): existing F32 block keep переносится к consumer, REPEAT/reshape/trim localCUDA3/4. Primary gates фиксируютpool1: PP98K1121.83->1154.72,+2.93% к pool1/block0; TG46.23->45.94,-0.62%singleton/no gain claim.24rawgroups/two98Kcorpora,NLLdelta0,4seedfault/retry,98operatorcases/fixed32capture и externalcompute failure/retry exact. No_alloc -520.52MiB@176K,actual176128/172032+16/repeat8K min1143.25MiB,all15360GPUbundles/routes/IDs exact,exit0.134widecompactkeep requests3.338GB заменилиfullmask26.700GB,fullptr/type/shape/stride/REPEAT/protectedslots0..3 proof; smallfallback прежний. По запросу пользователя LLAMA_DSV41_CANDIDATE_BLOCK_LOCAL unset/1ON,0OFF;фиксируется приcontextcreation.7freshno_allocdefault==explicit1checks passed,pool0nativeнеповторялся. Previouspool defaultOFF независим. Pool0native/F16KV/media/parallel/262K этим не объявляются проверенными.


## Текущие candidate defaults: native gate закрыт

[Отчёт](../experiments/1017-pool0-default-20261103/REPORT.md): runtime `my_maya 0e6f3cec3` и defaults не менялись. Pooling не задан (OFF); в default arm BLOCK_LOCAL также не задан (ON). На двух 98K text/code корпусах 24 группы logits/taps/state/rollback совпали побитово, delta NLL=0; четыре seed failures/retry прошли. Native OFF/default/OFF: 27 output arrays и draft counters совпали. PP98K 1107.75 -> 1138.16 t/s (+2.75%); TG 46.21 -> 45.82 (-0.84%, по одному измерению на arm, ускорение TG не установлено).

Actual ctx=163840 / prefix=159744 / repeat 8K прошёл: минимум 679.25 MiB свободной VRAM, все 15360 экспертных bundles на GPU, placement совпал, exit 0. Context=176128 не допущен расчётом при pooling OFF; прежний actual 176128 с pooling=1 остаётся отдельной серией. Это fixed Q3/MXFP8 Engram/Q8 KV/BF16 DSpark K3/P1 конфигурация, не универсальный предел контекста.

Ошибка post-test сборки сводки (`queue.py` перекрыл stdlib) вызвала повтор metadata. Исходные precision/native/payload manifests сохранены побайтно; память и routes metadata сверены с исходными admission inputs. Incomplete manifest, старые hashes и recovery diagnostics сохранены в отчёте. Runner переименован и защищён от выполнения при импорте.

Далее: matched Maya comparison, свежий residual profile, затем Q2/Q4 MMQ. Media, parallel>1 и auto-fit 262K остаются отдельными проверками. Нового commit/deploy нет.


## Новая платформа, 2026-11-05

[Новая независимая серия](../experiments/1017-maya-newhost-20261105/REPORT.md): Threadripper PRO 3945WX, 192 GB RAM, новый PCI/UUID-порядок GPU, CPU threads12. Все прежние timings и memory/cost estimates относятся к другой машине. Выполнено сравнение PP/TG без sidecar и с тем же sidecar на 512/8K/98K, с actual memory floor и fresh calibration. Результаты скорости, CPU, acceptance и repeat limitations приведены в отчёте. Далее свежий профиль prefill/verification; runtime/defaults не менялись.


## Новая платформа: подробная телеметрия Maya и нашего runtime

[Отчёт](../experiments/1017-maya-telemetry-20261105/REPORT.md): на записанном prefill8K условное число вычисляющих GPU Maya3.33, наше1.32;98K3.93/1.36. Polling/transport исключены, это не SM occupancy. Maya исполняет все40слоёв, наш pipeline enabled, но фактическое перекрытие слабее. TG profile указывает на target fragmentation и guard: anchor+5draft даёт6queries, KV-wide допускает6, обычный float/MoE batch guard5. Это кандидаты следующих native опытов, не реализованный фикс.

Профилировщик снижает наш TG примерно на27-29%, есть CUPTI GraphId diagnostics; CPU sampling/OSRT недоступны. Maya98K восстановлен после STOP/controller timeout без post-control; original manifest остаётся incomplete. Полный8K и оба наших захвата сохранены, runtime/defaults прежние. Нельзя выдавать recorded idle/API time за полную native critical-path decomposition. Следующий приоритет: bounded chunk/input/copy/feature lifetime для PP, counters-only sizes1..6 для verification.

## CPU lookup overlap, 2026-11-05: guarded default ON

[Report](../experiments/1017-cpu-lookup-overlap-20261105/REPORT.md). New host only: Q3_K-AntirezLike/MXFP8 Engram/Q8_0 KV/ctx100352/B8192/U512/P1/TopK2048/12 threads, all15360 target expert bundles on GPU, fixed GPU-only overrides and existing pipeline. BF16-named mixed MXFP4 DSpark, K5/full/p_min0.5 for speed. `GGML_SCHED_CPU_LOOKUP_EARLY=1` uses the existing protected CPU lookup staging; unsafe layouts fall back.

Clean native PP8K 1337.64 -> 1425.44 t/s (+6.56%); PP98K 1380.93 -> 1454.66 (+5.34%). TG improvement is not established; PP512 is within variation. 24 output/counter arrays exact, all three native processes exit0, sampled floor1271.25MiB. Two98K prose/C++ corpora yield12 paired precision groups exact including target/draft raw logits, taps37-39, sequence/global KV, restore, rollback and append shape changes; deltaNLL0 on128 positions per corpus, four seed failure/retry recoveries. External scheduler42 cases cover positive overlap and conservative fallbacks.

Initial tests used a private opt-in library. The user then requested guarded default ON and a commit; unset/1 enables, 0 disables. Promotion changes no isolation guards or arithmetic; its build/checks are separate in the report. Cache/staged MoE/CPUspill and direct host reads remain excluded; other models, media, parallel>1, larger context and real OOM/cancellation recovery are not certified by this gate. Residual prefill dependencies and six-query verification remain next.


## 2026-11-05 residual prefill, CPU lookup default ON

[Report](../experiments/1017-prefill-residual-20261105/REPORT.md): current f796e544f, same new-host Q3/MXFP8 Engram/Q8 KV/TopK2048/ctx100352/B8192/U512/P1/threads12/15360 GPU bundles/DSparkK5. Baseline host API probe closes8K/98K, then corrected CUDA13 attribution and bounded queue4x/copies8 screens. Queue/copy counts do not yield a useful PP gain and are not adopted. Actual minimum of copies8 is625.25MiB; no context reduction required. Shared gate/up6/29 co-location removed24 backward transitions but PP changed-0.65%, so it is not adopted. Default promotion, new arithmetic, full quality support, media/parallel and larger joint auto-fit are not granted by these diagnostic results.


## 2026-11-05 KV ownership audit and next design

[Audit](../experiments/1017-kv-owner-audit-20261105/REPORT.md): source20 CSA roots onCUDA2/3/4/5 are persistent Q8_0[512,100352,1,1], each54591488bytes. In the frozen layout, earlier LID fanout leaves source20 ROPE/Hadamard onCUDA4; actual root/copy lineage confirms the subsequent CSA write detour and first readers. Same copy1 row-index storage repeats at prefix0/2048. Five8K outputs/counters exact,42 external scheduler cases,374 split snapshots, minfree1323.25MiB,exit0; armed metadata PP is17.41% lower and not a native speed comparison.

[Approved prefill cache deferral](../designs/1017-prefill-cache-deferral.md) reuses the existing decode map with mandatory KV-only flush and initial opt-in/defaultOFF. The [implemented prototype](../experiments/1017-prefill-cache-deferral-20261105/REPORT.md) passes exact paired precision/state/rollback/NLL and actual ownership/flush checks; clean native OFF/ON/OFF completed: PP8K+4.45%, PP88K+7.40%, PP512-4.74%. Identicalctx92160/longprefix90112 is a fresh cohort after higher contexts failed conservative512MiB admission. Actual owner trace minimum587.25MiB; all15360 expert bundles remainGPU. Current defaults and production binaries remain unchanged.


## Approved prefill KV deferral: exact but memory-costly opt-in

[Completed evidence](../experiments/1017-prefill-cache-deferral-20261105/REPORT.md): same private module with flags0/1/0, ctx92160/longprefix90112, new-host fixed Q3/MXFP8 Engram/Q8 KV/DSparkK5/full/pmin0.5/U512/P1/15360GPU experts. PP512724.48->690.17t/s (-4.74%), PP8K1412.14->1474.92 (+4.45%), PP88K1449.66->1556.94 (+7.40%); TG gain not established. All27native output arrays/draft counters,12paired precision groups/two88K corpora/NLLdelta0/four failure-retries exact. Actual source20 producerCUDA4->CUDA2, pending KV-only flush and419raw/CSA/LID writes per arm verified. CUDA5 joint reserve+795.02MiB, native minimum535.25MiB; higherctx100352..94208 failed conservative admission. Flag LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1 opts in; unset/0 OFF. Code available for review, production modules/presets unchanged; no new commit/default promotion. Do not infer broad auto-fit262K/F16/media/parallel/CPU-cache support. Workspace and short-prompt regression are separate follow-up candidates; six-query verification remains independent.


## New-host verification width screen: six-query fragmentation measured

[Completed screen](../experiments/1017-verify-width-screen-20261105/REPORT.md), after runtime commit3854c3785: fixed prefix/tokens/state/n_rs_seq5, raw target decode+sync medians at8K:5queries43.03ms,6queries68.55ms (+59.30%);88K46.02->68.17ms (+48.12%). Separate lightweight counters show MoE120->720 and float275->950 matmuls, backend graphs44->114 /43->113. All156per-arm logits/splits match native/counter;24sampled state checkpoints per arm and base restores exact. Actual floor1391.125MiB, exits0, source/modules unchanged. Cold smaller-width graph instantiation rows are retained; measured width6has0instantiations and all llama graphs reuse. This is not total server TG/acceptance or a new implementation/default. [Next bounded plan](../designs/1017-six-query-moe-float.md): isolate grouped MoE6, then floating sample-grid6; shared-weight MMVF6 is a later separate kernel candidate.


## Six-query MoE5+1: scoped correctness evidence, default OFF

`LLAMA_DSV41_VERIFY_MOE_WIDE=1` is the committed opt-in for the bounded DeepSeek41 six-query/K5 path, with Q8 CSA/LID, text/P1, full CUDA expert placement/pipeline and separate contiguous Q3_K/Q4_K experts. Direct6 was rejected: Ampere Q3_K MUL_MAT_ID switches away from MMVQ above5, unlike Ada. The5+1 path leaves the CUDA selector and generic floating cap unchanged.

[Shared-prefix evidence](../experiments/1017-verify-shared-prefix-20261105/REPORT.md) passes117cases/18draft checks and observes MoE720->240/backend graph calls113->57, floating950 unchanged. [Nine raw OFF first8K captures](../experiments/1017-prefill-forensic-20261105/REPORT.md) and a [complete two-corpus conditional repeat](../experiments/1017-full-precision-forensic-20261105/REPORT.md) also pass, including restore/rollback and two failure/retries; actual floor1387.125MiB. NLLdelta0 uses the original4-query loop, while6query exactness has separate full-logit/state evidence. Earlier uninstrumented failures are unresolved. No default promotion, broad hardware/model coverage, native speed or production deployment is implied. Next is controlled cost/native measurement; mismatched controls must be retained and excluded from equivalent speed comparisons.
