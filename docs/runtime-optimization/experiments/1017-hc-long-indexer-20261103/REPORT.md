# HC default ON: actual176128 gate and indexer transfer audit, 2026-11-03

**Длинный memory/lifecycle gate пройден: ctx176128, полный172032+16, sampled minimum761.25MiB на каждой GPU не ниже512.** Следующий bounded кандидат подтверждён по размещению: полный F32 score20 передаётся ради pooling на другой GPU, затем большая dense candidate mask передаётся ещё раз. Подготовлен [placement-only design](../../designs/1017-indexer-placement.md); новый indexer runtime-код не реализован.

Base my_maya9b74cc6e6/runtime6d038260e плюс уже проверенный HC-sharing delta/defaultON. [Source/library/probe identity](artifacts/provenance.json), [runtime delta](artifacts/RUNTIME.patch). В этом этапе runtime source, defaults, пользовательский my_build/presets, GGUF и исходные веса не менялись. Нового commit/deploy/merge нет. Historical HC measured/promotion79 checksums сохранены.

## Frozen fixture

Q3_K-AntirezLike10 shards, две MXFP8 Engram lazy CPU/mmap; Q8_0K/V, index_topk2048. BF16 DSpark K3/full/p_min0.5/backend samplingON; target all15360 routed bundles GPU, cache/staging/fit/mmprojOFF. B8192/U512/P1/20threads, tensor-split7,8,8,7,8,3 и прежние exact tensor overrides; draftCUDA5/CUDA0. HC sharing/M2/producer-LOCAL flags unset, ASYNC unset/defaultOFF. Полные argv/env, raw IDs, PCI/UUID mapping и actual per-tensor routes сохранены в artifacts. Это fixed-layout проверка, не доказательство auto-fit262K или поддержки media/parallel>1.

## Actual context176128

Перед payload пересобраны metadata/loader-only probes с текущими internal headers/libs. Fresh no_alloc100352/default,176128/default/1,163840/default,159744/default. Default/explicit1 budgets и routes exact. Допуск176128 рассчитан по max observed per-device overhead текущего HC-ON100352 и исторического M2/copyOFF176128, плюс64MiB guard: guarded minimum694.27MiB. Контекст уменьшать не потребовалось.

В отдельном реальном процессе:8192+256,172032+16,8192+16, все cache_n0. Target placement exact к metadata,15360 GPU complete bundles, host routed bytes0. Loader observer читает только CPU model metadata, не tensor payload; forward interposer/profiler/eval callback и дополнительные probe GPU buffers отсутствуют. Startup ready minimum905.25MiB.

| Logical CUDA | NVML index | Load min free, MiB | Inference min free, MiB |
| --- | ---: | ---: | ---: |
| CUDA0 | 1 | 1351.25 | 1145.25 |
| CUDA1 | 2 | 2135.25 | 1995.25 |
| CUDA2 | 4 | 2311.25 | 2179.25 |
| CUDA3 | 0 | 969.25 | 775.25 |
| CUDA4 | 5 | 905.25 | 761.25 |
| CUDA5 | 3 | 1231.38 | 1143.38 |

NVML sampler100ms, owned-process guard200ms; это sampled floor, не гарантия отсутствия пика между samples. Global minimum761.25MiB, exit0, без floor breach/OOM/CUDA error/compute-size mismatch/forced kill или host feature-packing fallback. LOCAL активен.8K256 token array exact к prior native benchmark;172K16 exact к prior M2/copyOFF176128; повторный8K16 exact к первым16 текущего8K. Отдельный NLL набор не повторяли: arithmetic/transport implementation после прежнего full exact gate не менялись.172K input - повтор frozen98304 IDs для memory/lifecycle stress, не корпус оценки качества. Одиночные timings payload не являются новым speed A/B.

[Memory summary](artifacts/memory-summary.json), [полный payload](artifacts/payload-176128/results.json), [admission](artifacts/payload-admission.json).

## Separate diagnostic process ctx100352

Nsight2025.3.2 плюс read-only LD_PRELOAD observer: CPU node/source metadata (ne/nb/type/root/view/offset/data/buffer), NVTX вокруг существующих backend/CUDA-op calls, без eval callback, backend iface mutation или GPU buffer/payload reads. Аргументы/streams исходных функций передаются без изменения. Единственная дополнительная probe CUDA synchronize - помеченный profiler drain после HTTP response, вне prompt timer.

Один512 smoke и четыре8K, четыре98K запроса warm/pre/profile/post,16greedy/seed1234/ignore_eos. Все8 массивов tokens и draft generated/accepted exact к прежним native16-token controls, не только друг к другу; все cache_n0. M2 и tail-seeding сохранились, actual placement exact к metadata, minfree1501.25MiB,exit0. Сборка не менялась.

| Prefix | Pre-control PP time, ms | Capture PP time, ms | Post-control PP time, ms | Target ubatches / KV-only | Draft seed rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 7484.89 | 9044.48 | 7536.34 | 17 / 8 | 1536 |
| 98304 | 88853.55 | 90300.00 | 89080.26 | 193 / 107 | 12800 |

Контроли - диагностический процесс с dormant interposer/Nsight, не новая чистая A/B серия. Чистый HC OFF/ON/OFF результат остаётся в [предыдущем отчёте](../1017-hc-copy-sharing-20261103/REPORT.md).16 выходных токенов не используются для новой оценки TG.

## Фактический indexer dataflow

Source20, block8, top2048 **блоков**, final2048 **позиций**. Это разные selectors: candidate set может содержать до16384 rows. Source20 имеет compression_ratio1; нельзя делить его history ещё на4. Sources2/8/14 имеют ratio2. На98K idx_score20 выполняется86 раз, но candidate branch только72 раза: M2 удаляет ненужную suffix-работу в107 из193 target ubatches; ещё14 score20 frames не нуждаются в block pruning. На8K candidate branch и соответствующие cross-GPU copies отсутствуют.

| Узел | Backend | Реальный consumer / форма |
| --- | --- | --- |
| idx_score-20 / его final TopK | CUDA2 | LIGHTNING_INDEXER F32 score и TopK2048 |
| idx_score-20 (cont) | CUDA3 | Полный scheduler input CUDA3#idx_score-20#slot, затем CONT |
| candidate_block_score-20 | CUDA3 | POOL_2D max8, затем pin/add/TopK/scatter |
| candidate_mask-20 | CUDA3 | REPEAT/RESHAPE, затем ADD к independently computed scores24/28 |
| CUDA4#candidate_mask-20#slot | CUDA4 | ADD к scores32/36; промежуточные layers переиспользуют их final TopK |

В72 frames обе большие формы равны: [n_pos,512,1,1] F32, максимум[98304,512,1,1] =192MiB; contiguous/direct или zero-offset full reshape. Pooled form [12288,512,1,1] максимум24MiB. В captured fixture padding0, но новый дизайн обязан сохранить partial-block path. Candidate pin - F32 CPU input с+inf в newest visible block; matched uploads72,1.098GB. Сам CPU fill не измерен отдельным timer, его wall cost не выводится из upload API.

POOL_2D поддерживается CUDA в current source. Candidate score/CONT/pool не имеют layer-local assignment hint в graph_get_cb, в отличие от HC/index-prefix ops; scheduler расширяет соседние assignments. Метаданные подтверждают фактическое размещение, но не записывают внутренний SET_CAUSE для каждого узла. Поэтому причина формулируется как кандидат улучшения placement, не доказанный missing-support bug.

## Copy cost: границы доказательства

Scheduler вызывает backend iface.cpy_tensor_async напрямую. Generic ggml_backend_tensor_copy_async wrapper этих inputs не видит (0 ix_copy_metadata); его пустой счётчик не означает отсутствие transfers. Использована отдельная корреляция: конкретный cloned input/name/full shape/backend у следующего same-thread destination graph внутри того же ubatch, единственная подходящая категория/size, затем CUDA API correlationId к CUPTI records. Совпали все72 score и72 mask frames. На каждой записи есть ожидаемые D2H source-device и H2D destination-device. Src/dst pointer аргументы CUDA API **не перехватывались**: это metadata-constrained attribution, не pointer-level proof. Неоднозначные/unmatched calls не включаются.

| Edge | API calls | Logical payload, GB | Inclusive host API sum, ms | CUPTI records | Bytes в records, GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| idx_score-20, CUDA2->3 | 72 | 8.782 | 251.15 | 144 | 17.564 |
| candidate_mask-20, CUDA3->4 | 72 | 8.782 | 1045.24 | 144 | 17.564 |

У каждого API по две CUPTI activity records kind2/kind1 с полным logical size, всего35.127GB record bytes для двух edges. Это не аппаратный счётчик байтов PCIe. CUPTI duration sums23.30/34.15ms не используются как пропускная способность или полная цена DMA; они явно не описывают весь transfer/wait path. Inclusive host API1.296s тоже включает возможное ожидание и не является critical-path saving.72 interposed pool launches коррелируются с72 CUDA kernels, kernel sum11.93ms; это только pool, не весь block selector/CONT/expansion.

Nsight DIAGNOSTIC_EVENT содержит предупреждения о неполном сборе NVTX/CUDA events и ошибки GetGraphId/GetGraphNodeId: по2+2 в8K,1800+1800 в98K. Они сохранены в copy-correlation.json. Выбранные144 copy requests/288 records и72pool kernels имеют полное соответствие ожидаемому metadata count, однако полная graph replay/kernel critical-path decomposition этим capture **не закрыта**. Нельзя складывать эти sums и объявлять ожидаемый PP gain. Перед реализацией не требуется придумывать новый attention kernel; сначала проверить один placement-only вариант чистым A/B.

[Raw analysis](artifacts/indexer-analysis.json), [copy correlation/matching и tool diagnostics](artifacts/copy-correlation.json), [validation](artifacts/validation.json). SQLite/nsys-rep остаются в/tmp, hashes/locators в provenance.json; machine-readable extracts и raw observer/server logs сохранены в Git-visible artifacts.

## Следующий шаг

[Конкретный design](../../designs/1017-indexer-placement.md): сохранить все existing ops/dtypes/численную семантику, перенести **CONT + padding/CONCAT + pooling** на подтверждённого score producer. Тогда первый edge должен переносить block scores8x меньше. Pinning только pool не годится: CONT может остаться на CUDA3. Не удалять CONT и не менять block/final TopK в том же эксперименте. Первый prototype opt-in/defaultOFF с existing graph/backend ownership и fallback; после него exact logits/state/rollback, actual memory floor и чистая native OFF/ON/OFF PP512/8K/98K/TG.

Вторая отдельная гипотеза: передавать block keep и разворачивать dense mask локально у actual consumers. Для неё нужен отдельный multi-device lifetime design. Ни новый indexer default, ни этот runtime prototype в данном этапе не вводились. Matched Maya, Q2/Q4MMQ и другие режимы остаются отдельными пунктами плана.
