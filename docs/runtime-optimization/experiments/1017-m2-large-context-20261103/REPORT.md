# M2: actual ctx176128 и аудит view transfers, 2026-11-03

**Больший-context memory gate пройден. Подтверждены повторные CUDA API copy requests для HC tensor и его reshape на двух границах GPU.** Runtime `my_maya 828ee5b94`, implementation `6d038260e`; исходники, defaults и пользовательский `my_build` не менялись. Это capacity/lifecycle gate и read-only диагностика, без нового speed A/B или сравнения с Maya.

## Условия

Q3_K-AntirezLike, две MXFP8 Engram, Q8_0 K/V, BF16 DSpark K3/full/p-min0.5/backend samplingON, TopK2048. B8192/U512/P1/20threads, frozen split/overrides, все15360 routed expert bundles GPU, fit/cache/staging/mmprojOFF. `LLAMA_DSV41_PREFILL_SKIP`, GPU_FEATURES и LOCAL не заданы: используются текущие ON defaults; ASYNC unset/OFF. Не переключали precision или placement ради прохождения теста.

[Expanded argv/environment и library hashes](artifacts/config.json), [CUDA ordinal/NVML PCI-UUID mapping](artifacts/gpu-map.json), [provenance](artifacts/provenance.json). NVML0 не равен CUDA0; floor проверялся по UUID каждой карты.

## Свежая metadata/admission проверка

Metadata helper, loader observer и API observer пересобраны с текущими headers/libraries до загрузки payload. Пять no_alloc cases:100352/default,176128/default,176128/OFF,163840/default,159744/default; все exit0. Target и связанный draft/taps учитываются вместе. Model/context/compute budgets и per-tensor routes176128/default и OFF совпали точно. M2 сокращает работу, но сохраняет полный fallback reserve: экономии persistent/compute allocations от M2 здесь не заявляется.

Это fixed-layout `measure` с no_alloc/load_modeNONE, а не свободный auto-fit solver. Для допуска взят максимум observed per-device overhead из свежего M2/native100352 и двух historical176128 controls, затем дополнительный guard64MiB. Ни один источник не подменяет actual payload. Минимальные guarded estimates:176128:537.25MiB,163840:672.12MiB,159744:720.90MiB. Все прошли512MiB; выбран176128, уменьшение не потребовалось. [Исходные расчёты и hashes reference JSON](artifacts/payload-admission.json).

## Actual payload и завершение

Один собственный server process, context176128. Последовательно выполнены:

| Full uncached prompt | Outputs | Проверка |
| ---: | ---: | --- |
| 8192 | 256 | Все256 IDs совпали с прежним M2 benchmark |
| 172032 | 16 | Весь prefix пересчитан; IDs совпали с historical pre-M2 LOCAL176128 на том же prefix |
| 8192 | 16 | После длинного запроса, с включённым observer; IDs совпали с первыми16 из текущего8K+256 |

Везде cache_n0, greedy/seed1234/ignore_eos, target/draft активны. В логе подтверждены M2 `prefill_skip: source=20` и `dspark_prefill_tail`. Аудит armed только перед третьим запросом. После старта запас745.25MiB, minimum за load/inference601.25MiB. Actual per-tensor placement exact к свежей metadata; все15360 expert bundles/220.331GiB routed weights GPU, routed host bytes0.

| Backend | NVML index | Metadata projected free, MiB | Actual inference sampled min free, MiB |
| --- | ---: | ---: | ---: |
| CUDA0 | 1 | 1424.28 | 1145.25 |
| CUDA1 | 2 | 2045.99 | 1835.25 |
| CUDA2 | 4 | 2220.63 | 2019.25 |
| CUDA3 | 0 | 876.32 | 615.25 |
| CUDA4 | 5 | 820.06 | 601.25 |
| CUDA5 | 3 | 1296.69 | 1143.38 |

NVML sampling100ms, watcher200ms останавливает только собственный процесс при cumulative sampled floor<512MiB. Watcher не сработал. Это sampled peak, не гарантия отсутствия краткого пика между samples. Нет OOM/CUDA errors/compute-size mismatch, forced kill или host packing fallback. Server exit0, после опыта compute apps отсутствуют, все GPU освобождены.

Это один проход memory/lifecycle stress с dormant interposer, не clean native speed benchmark: LD_PRELOAD scheduler/CUDA wrappers загружены во всех трёх запросах, но до audit.arm сразу передают вызовы дальше. Observer не добавляет GPU allocations, payload reads, streams, synchronization или eval/abort callbacks. CPU metadata/calibration/compile параллельно inference не запускались. Одиночные timing сохранены в raw results; PP/TG gain и acceptance improvement не выводятся из этой проверки.

### Ошибка постобработки стенда

Все три запроса и штатное завершение server завершились, но driver получил exit1 при заключительном сравнении historical JSON: использован `prompt_n` вместо `n`. [Raw results](artifacts/payload-176128/results.json) сохраняют complete=false/error; они не переписаны как успешные. [Отдельный валидатор](artifacts/validate-completed.py) исправляет чтение схемы и подтверждает все responses, hashes, placement, memory/exit и три ID comparisons: [validation.json](artifacts/validation.json), complete=true.

Начальный загруженный Python source сохранён как run-payload.initial.py. Исправленный run-payload.py записан во время выполнения: traceback может показывать новую дисковую строку при старом KeyError. Повторный GPU-тест ради исправления schema comparison не запускался. Raw server.log содержит неполные UTF-8 fragments tokenizer metadata; текстовые поиски используют errors=replace, исходные bytes сохранены.

## Что доказал аудит копий

Один8K+16 request,16 wide target scheduler frames:15x512 rows и1x508, всего8188 rows. Последние4 rows и короткая generation/draft не входят в этот аудит. Каждый frame завершился status0, адреса/размеры выбранных tensor metadata после вызова прежние.

Для каждой из двух границ на каждом frame найдены **две успешные `cudaMemcpyPeerAsync`** с одинаковыми source address, byte count, source/destination devices и stream, но разными destination addresses. Source metadata: общий root/data, view_offs0, contiguous F32 и равные полные byte ranges. Destination pointers соответствуют двум отдельным scheduler copy tensors; потребители каждого найдены в том же графе.

| Source pair | Edge | Frames | Full-range API requests | Потенциально устранимый повторный запрос, bytes |
| --- | --- | ---: | ---: | ---: |
| l_last-6 / reshaped | CUDA0->CUDA1 | 16 | 32 | 670760960 |
| l_last-14 / reshaped | CUDA1->CUDA2 | 16 | 32 | 670760960 |
| Итого | | 32 пары | 64 | 1341521920 |

Это1.342GB/1.249GiB повторно запрошенных bytes в пределах выбранных двух full-range pairs. Это не общий transfer traffic, не CUPTI physical DMA count и не измеренная экономия wall time. Host-staged исполнение peer-copy возможно, связь с физическими copy engines в этом опыте не снималась.

Пример первого frame, граница6: root `[5120,4,512,1]`, reshape `[20480,512,1,1]`, оба41943040bytes/40MiB по source0x7e71a8f2c000. Два destination0x7e72841c7300 и0x7e7286bcc300. Первый используется hc_attn_pre-7 и hc_attn_post-7; второй - RMS_NORM для HC mixes. На границе14 аналогичные consumers слоя15. Полные ne/nb, root/view pointers и consumer lists сохранены в [audit-analysis](artifacts/audit-analysis.json) и [raw frames/API JSONL](artifacts/payload-176128/transfers.jsonl).

Нельзя приписывать все187 API calls, пересекающихся с выбранными addresses, этим HC tensors: allocator повторно использует тот же диапазон для других временных результатов. В статистику выше попали только32 пары с полным размером, точной root/view alias-связью, destination metadata и consumers;12 partial-range groups исключены. Это существенная граница pointer-only диагностики.

[Static source audit](artifacts/source-audit.json): scheduler keyed copy по hash конкретного src tensor, поэтому root и reshape получают разные copy slots. CUDA backend отправляет memcpy для каждого принятого input. После объединения destinations нужно заново доказать allocator/events/lifetime и fusion safety; равные source bytes сами по себе этого не разрешают. Здесь нет runtime fix.

## Решение и следующий шаг

- Greater-context M2 fixed-placement memory/lifecycle gate176128 закрыт с floor512MiB; прежний pre-M2 результат больше не единственное основание для этого контекста.
- Следующий bounded кандидат - одна destination copy для zero-offset contiguous HC root и его reshape на том же backend/cur_copy. Сначала согласовать отдельный design с ownership/events, destination views, source/destination lifetime и graph/fusion guards; никаких дедупликаций только по равенству addresses/имён.
- До promotion нужны operator/negative controls для offset/noncontiguous/different type/device, exact target/draft logits/state/rollback/reuse, no_alloc/actual512MiB и native OFF/ON/OFF PP512/8K/98K/TG. Physical transfer correlation и текущий timing baseline должны предшествовать выводу о скорости. Сохранить ASYNC/media/parallel и иных unsupported режимов fallback.
- Dense idx_score/candidate_mask traffic, новый matched Maya, Q2/Q4 MMQ и auto-fit262K остаются отдельными задачами. Их прохождение не выводится из current176128 text-only fixture.

Все новые результаты сохранены в базе знаний ветки my_maya. Нового коммита и изменений runtime/defaults нет. [Воспроизведение](artifacts/REPRODUCE.md), [контрольные суммы](artifacts/SHA256.json).
