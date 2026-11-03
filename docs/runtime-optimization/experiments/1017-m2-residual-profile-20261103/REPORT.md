# Residual prefill после M2, 2026-11-03

Этап завершён на `my_maya 828ee5b94`, runtime implementation `6d038260e`. Default M2 включён, флаг `LLAMA_DSV41_PREFILL_SKIP` не задан. Это новый профиль оставшейся работы и проверка гипотезы ожидания input copies. Runtime, defaults, модель и пользовательский `my_build` не менялись. Новый speed gain или matched Maya comparison здесь не заявляются.

## Условия и контроли

Тот же Q3_K-AntirezLike/MXFP8 Engram/Q8_0 KV/BF16 DSpark K3/full/p-min0.5/backend samplingON, default LOCAL/ASYNCoff. Context100352/B8192/U512/P1/TopK2048,20threads, frozen all15360 GPU expert bundles, прежние split/overrides, fit/cache/staging/mmprojOFF. Изменять precision/placement между controls не требовалось.

Сначала чистый native process:512/8K/98K,warm+1control. Затем отдельный NSYS process: smoke512/control+capture;8K/98K,warm/pre-control/capture/post-control. Всего16 запросов, каждый пересчитывает полный prompt/cache_n0,16 greedy outputs, одинаковые raw IDs/seed/penalties/stop policy. NSYS: CUDA+NVTX,node trace, CPU samplingOFF; eval/abort callbacks отсутствуют. Observer пересобран с актуальными headers и учитывает новый member decode/KV-only argument, чтобы видеть internal seed injections. Контроли вне capture всё ещё выполняются под Nsight/interposer.

| Prefix | Чистый native control PP,t/s | NSYS pre/post PP,t/s | Capture PP,t/s | Capture overhead к median pre/post |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1050.63 | 1037.31..1046.94 | 1020.51 | +2.12% |
| 98304 | 1069.16 | 1062.84..1064.66 | 1053.61 | +0.96% |

Native control только один после прогрева, не новая статистическая серия A/B. Smoke512 служит проверке observer и не используется для вывода о скорости. Эти16 outputs не заменяют прошлый TG256. Все output arrays и draft generated/accepted exact между native, NSYS controls и captures для своего prefix. Все kernels/memcpy сопоставлены launch API, unmapped0.

Все процессы exit0, без OOM/CUDA error/compute-size mismatch/forced kill. Actual sampled minimum free за load/inference: native1509.25MiB, NSYS1495.25MiB; floor512 выдержан. GPU освобождены. Sampling100ms и stop watcher200ms не являются гарантией отсутствия пика между samples.

## Остаточная GPU-работа

Суммы длительностей target prefill kernels, секунды:

| Семейство | 8K | 98K |
| --- | ---: | ---: |
| Routed Q3/Q4 matmul | 3.989 | 47.166 |
| Attention kernels | 1.592 | 16.330 |
| Named indexer kernels | 0.058 | 5.960 |
| Other quant matmul | 0.505 | 5.778 |
| Все target kernels | 7.058 | 86.712 |

Routed Q3/Q4 занимает56.5%/54.4% суммы kernels; attention22.6%/18.8%. Классификация по demangled symbols: projections/norms индексатора могут входить в другие семейства, поэтому Named indexer не равен всей стоимости indexer subgraph. Q3 kernel type11/tile32 составляет43.299s из47.166s routed суммы на98K. Это подтверждает, что Q3 остаётся главным compute кандидатом; прежний проигравший serial FP16/GEMM не получает нового разрешения из этой цифры.

Суммы kernels на разных GPU перекрываются и не являются процентами PP wall time. По target-tagged kernel+copy unions среднее число одновременно активных устройств, пока есть target GPU activity:1.26 на8K,1.32 на98K. Это не SM occupancy и не утверждение, что все шесть карт могут быть полезно заняты в каждом M2 KV-only chunk.

CUPTI device IDs сопоставлены с backend CUDA по enclosing NVTX launches для всех6 устройств; здесь deviceId совпал с backend ordinal. `TARGET_INFO_GPU.id` имеет другой порядок: следует использовать `cuDevice`/PCI/UUID mapping в gaps.json, а не считать raw profiler GPU id CUDA ordinal.

## CPU lookup и настоящий draft seed

На98K193 target ubatches, в каждом три CPU графа: BF16 token GET_ROWS и MXFP8 Engram1/14. Суммарно2.027s:0.121s token embeddings,0.946/0.960s две Engram lookup группы. На8K17 ubatches,0.161s суммарно. Даже идеальное сокращение этого CPU участка ограничено его вкладом и перекрытием, не обещает пропорционального PP gain.

M2 реально активен: capture8K восемь KV-only ubatches,98K107. В production request8K не один logical8192 decode: checkpoint manager использует7676+512+4, поэтому реальных draft seed rows1536, а не1024 из direct8192 fixture. На98K seed rows12800. Размер HTTP prompt нельзя подставлять вместо logical/ubatch shapes.

На98K draft seed kernel sum18.992ms, CPU decode spans32.164ms. Inclusive bridge12.393s содержит feature_get_packed12.148s, включая ожидание ещё выполняющегося target. После вычитания packed getter и decode children остаётся212.351ms bridge setup/bookkeeping/metadata, а не12s работы draft. Pack compute425.209ms также включает copies/waits; kernel-only packing1.731ms. Эти вложенные времена нельзя складывать.

## Где остаются ожидания

В NVTX prompt-span98K длиной93.187s union зарегистрированных kernels/memcpy/memset занимает70.623s. В оставшихся22.564s не записано GPU activity; это определение по trace, не доказательство физического простоя всех GPU engines.

Из этих интервалов19.849s пересекаются с runtime CUDA API: cudaEventSynchronize7.088s, cudaStreamSynchronize6.158s, cudaLaunchKernel4.188s, cudaMemcpyPeerAsync1.969s. Это интервальные пересечения, не суммы всех inclusive API durations и не доказательство, что fence можно удалить. CPU graph intersection1.412s. На8K соответствующий no-recorded-GPU interval1.860s. Call stacks не снимались; подмена attribution точной причиной недопустима.

### Дополнительная проверка input-copy гипотезы

Отдельный третий process с существующими LLAMA_GRAPH_TIMINGS/GGML_SCHED_TIMINGS/GGML_SCHED_COPY_STATS, без Nsight/interposer/callbacks. Три холодных512/8K/98K запроса по16 outputs, не speed A/B. Outputs/draft counters exact к native; exit0/minfree1517.25MiB, GPU освобождены.

На98K192 wide target rebuilds. Их build124.232ms, alloc615.841ms, весь rebuild893.676ms. Внутренний sched alloc_graph timer покрывает612.470ms; разность с outer alloc составляет3.371ms. Эта разность включает wait_input_copy до внутреннего timer и call/log overhead; это не отдельный wait-only timer. Прежний compute_splits input_wait_us во всём запросе18.882ms, input_copy1.486ms; async ветви имеют иную семантику счётчика. Наличие48s inclusive cudaEventSynchronize из NSYS поэтому не объясняется узким measured alloc input-copy участком. Расширять число copies4->8 только по этим данным не обосновано.

Реальный guard перед overwrite остаётся необходимым. Более широкую причину waits нужно локализовать по source/target, compute/copy dependencies и live buffers; простое удаление synchronize или повтор старого FFN-only region experiment не является результатом этого этапа.

## Межкарточные данные: следующий ограниченный аудит

NSYS target prefill98K: H2D133.921GB/4.827s copy-engine sum, D2H120.076GB/1.670s, P2P37.312GB/0.065s. Это физические trace records; host staging peer traffic нельзя объявлять CPU experts или передачей target features. В fixture все эксперты GPU-resident.

Scheduler follow-up насчитал130.356GB logical GPU->GPU mode=async,7.864GB graph_input_async и7.455GB host_stage_async. Это cumulative counters по всем трём diagnostic requests, включая generation и оба contexts; они не равны physical98K copy bytes. Sync fallback лишь1.155MB, так что большой host traffic возможен даже при scheduler async branch.

Крупнейшие logical edges из follow-up:

| Edge/tensor | Count | GB |
| --- | ---: | ---: |
| CUDA2->3 idx_score-20 | 78 | 8.788 |
| CUDA3->4 candidate_mask-20 | 78 | 8.788 |
| CUDA0->1 l_last-6 | 232 | 8.771 |
| CUDA0->1 l_last-6 (reshaped) | 232 | 8.771 |
| CUDA1->2 l_last-14 | 232 | 8.771 |
| CUDA1->2 l_last-14 (reshaped) | 232 | 8.771 |

Уже видны transfer pairs исходного HC tensor и reshape с одинаковыми count/bytes, а frozen partial-FFN overrides создают дополнительные crossings swiglu/gate projections. Это кандидат на проверку duplication и placement, не доказанный лишний DMA: до фикса нужны actual root/data/view offsets/strides/byte ranges, copy call IDs и consumers/lifetime. Статистика не разрешает автоматически снимать overrides или менять layouts, обеспечивающие VRAM fit.

Следующий bounded аудит: для одного такого boundary подтвердить alias/read-only/contiguous ranges и связь с actual transfers; затем описать, как одна destination copy могла бы обслуживать оба views с прежними dependency/allocator/graph-property guards. Отдельно проверить плотный candidate-score/mask traffic. До нового дизайна и correctness/native gates source не менять. Больший-context M2 memory/fit gate остаётся следующим пунктом основного плана; Q2/Q4 MMQ и свежий matched Maya - отдельные этапы.

## Артефакты и ограничения

[Summary](artifacts/summary.json), [raw requests/configs/binary hashes](artifacts/results.json), [analysis](artifacts/analysis.json), [GPU/API intervals и mapping](artifacts/gaps.json), [scheduler follow-up](artifacts/dispatch/summary.json), [источники и воспроизведение](artifacts/REPRODUCE.md), [raw trace locators/hashes](artifacts/trace-locators.json), [контрольные суммы](artifacts/SHA256.json).

Изменения этого этапа - reports/knowledge/plans и внешние probes. Новых kernels/defaults/model conversions не было. NLL/rollback/новый context176128 здесь повторно не тестировались: numerical gates M2 сохранены отдельно. Один technical-text workload и короткие controls ограничивают выводы; неподдержанные multimodal/parallel/LOCAL+ASYNC и arbitrary OOM/cancel не закрыты.
