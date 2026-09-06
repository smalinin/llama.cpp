# Отчет о локальных изменениях GLM5NEXT

## Область отчета

- Ветка: `glm5next-upstream-optimized`
- Upstream для сравнения: `origin/glm5next/upstream`
- Общий предок: `f30bed88717059d8a4728864c88f8abad8d329a0`
- Коммитов с реализацией поверх upstream: 31
- Порядок ниже: хронологический, от первого изменения к последнему

Документационные коммиты, изменяющие только этот отчет и его английскую
версию, в список реализации не включаются.

Список можно воспроизвести командой:

```bash
git log --reverse --oneline origin/glm5next/upstream..HEAD -- . \
  ':(exclude)GLM5NEXT_LOCAL_CHANGES.md' \
  ':(exclude)GLM5NEXT_LOCAL_CHANGES_EN.md'
```

## Краткий итог

Работа была разделена на несколько направлений:

1. Сокращение стоимости sparse attention при длинном контексте.
2. Persistent cache для завершенных indexer pool keys.
3. Полная поддержка MTP, включая prompt reuse, мультимодальность и несколько slots.
4. Устранение CPU-копий hidden states и embeddings.
5. Исправление auto-fit, LCP rewind, multi-slot rebuild и размеров инкрементальных обновлений.
6. Оптимизация indexed FlashAttention, Lightning Indexer и CUDA Graph.
7. Оптимизация MoE decode и fused expert down reduction для Q3_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS.

## Полный список коммитов

### 1. `b888281` - `glm5next: compact sparse decode attention`

- Что сделано: decode attention переведен на компактный набор из выбранных pool members, незавершенного tail и выравнивающего padding вместо обработки полного KV диапазона.
- Зачем: ограничить объем attention выбранными токенами и уменьшить падение скорости генерации при росте контекста.

### 2. `7588579` - `glm5next: cache completed pool keys`

- Что сделано: добавлено постоянное хранение сжатых ключей завершенных pools и инкрементальное обновление только новых или invalidated записей.
- Зачем: не пересчитывать все ранее завершенные pool keys на каждом decode-шаге; стоимость компрессии больше не должна расти вместе со всем контекстом.

### 3. `f947340` - `glm5next: add full MTP decoding`

- Что сделано: добавлен полноценный MTP draft context, передача hidden states, управление MTP KV/recurrent/pool cache и интеграция с speculative decoding.
- Зачем: использовать встроенную next-token голову GLM-5.3-Flash для speculative decoding без отдельной draft-модели.

### 4. `fb88d4a` - `glm5next: skip empty recurrent MTP cache`

- Что сделано: операции recurrent MTP cache пропускаются, когда для последовательности еще нет состояния.
- Зачем: исключить обращения к пустому cache и ошибки на начальных или сброшенных MTP-последовательностях.

### 5. `e467c0b` - `glm5next: handle empty recurrent MTP graph input`

- Что сделано: graph input для recurrent MTP корректно обрабатывает отсутствие входного состояния.
- Зачем: дополнить предыдущий guard на уровне графа и устранить ошибки построения/выполнения при пустом MTP state.

### 6. `3e789bb` - `fit: track shared MTP target placement`

- Что сделано: auto-fit учитывает размещение target и встроенных nextn/MTP слоев при оценке общей модели и контекста.
- Зачем: не получать ложную оценку свободной VRAM и последующий OOM при создании MTP context на большой длине контекста.

### 7. `4edb38e` - `cuda: accelerate GLM lightning top-k selection`

- Что сделано: добавлен специализированный CUDA top-k путь для формы данных GLM Lightning Indexer.
- Зачем: уменьшить время выбора pools, которое становилось заметной частью decode при длинном контексте.

### 8. `aa567f5` - `server: keep MTP state aligned across prompt reuse`

- Что сделано: target, draft и speculative state синхронизируются при LCP reuse, rewind и повторной обработке prompt; добавлен безопасный full-reprocess fallback.
- Зачем: устранить non-consecutive position warnings, расхождение позиций draft/target и ошибки `ctx_dft cannot catch up`.

### 9. `26cf4fe` - `server: suspend MTP for multimodal requests`

- Что сделано: MTP временно отключался для запросов с media embeddings.
- Зачем: сохранить корректность до появления полной синхронизации мультимодального target/draft пути.
- Статус: это защитное ограничение позднее заменено полноценной поддержкой в `cfcdf4e`.

### 10. `7ac7124` - `glm5next: accelerate long-context sparse prefill`

- Что сделано: добавлен indexed FlashAttention, который читает K/V напрямую по выбранным индексам, а также CPU/CUDA инфраструктура индексов и новый выбор dense/indexed пути.
- Зачем: не создавать и не обрабатывать полный dense attention диапазон для каждого prefill token при большом KV cache.

### 11. `fcd2441` - `glm5next: keep small MTP batches off indexed attention`

- Что сделано: маленькие speculative batches оставлены на dense attention пути; indexed attention включается только при достаточном размере batch/KV.
- Зачем: избежать провала производительности около длинного контекста, когда indexed kernel не получает достаточно параллельной работы.

### 12. `cfcdf4e` - `glm5next: enable multimodal MTP fast path`

- Что сделано: MTP синхронизирован с media batches, vision embeddings передаются напрямую между CUDA backends, а совместимые vision graphs кешируются.
- Зачем: после мультимодального prefill сохранить ускорение генерации от MTP и убрать лишние GPU-CPU-GPU копии и повторную сборку vision graph.

### 13. `1315090` - `glm5next: accelerate long-context MTP attention`

- Что сделано: indexed attention распараллелен между несколькими warps; добавлено повторное использование indexer/pool state между связанными MTP итерациями и проверка совместимости graph inputs.
- Зачем: ускорить маленькие MTP batches на большом KV cache, где один warp и повторная подготовка индексов становились узким местом.

### 14. `99dc6b4` - `glm5next: align metadata and quantization handling`

- Что сделано: синхронизированы GGUF metadata keys, tokenizer fallback/prefix, флаг совместного использования indexer в MTP, vision SwiGLU clamp key и правила quantization.
- Зачем: корректно конвертировать и загружать новые GLM-5.3 файлы, не теряя параметры архитектуры и не квантуя несовместимые tensors.

### 15. `4a7b87f` - `perf(glm5): preserve MTP pool cache across draft steps`

- Что сделано: добавлен Q8_0 indexed CUDA attention; завершенные pool keys сохраняются между MTP steps, а invalidation применяется только к затронутым speculative blocks.
- Зачем: не сбрасывать полезный persistent cache после каждого draft/verify цикла и ускорить indexed attention с Q8 KV cache.

### 16. `1f0fccd` - `fix(glm5): size pool updates for pending dirty keys`

- Что сделано: емкость incremental pool updates теперь учитывает все pending dirty keys, включая изменения после MTP и LCP rewind.
- Зачем: предотвратить переполнение update capacity, ошибку graph reuse и падение после отката speculative state.

### 17. `40efd56` - `cuda: optimize GLM sparse attention and MoE decode`

- Что сделано: перенесены и адаптированы fused SwiGLU clamp, multi-token MoE kernels, weighted expert reduction, FlashAttention swizzling и multi-GPU graph optimization; sparse MMA attention адаптирован к compact-index пути GLM.
- Зачем: получить оптимизации новой CUDA/MoE реализации, сохранив совместимость с локальным sparse attention и multi-GPU конфигурацией.

### 18. `155c6a1` - `cuda: avoid weighted MoE fusion for single-token decode`

- Что сделано: общий fused weighted-expert reduction оставлен для prefill и verify batches, но отключен для обычного single-token decode.
- Зачем: измерения показали, что на одном токене старый vector reduction быстрее общего multi-token fusion.

### 19. `7e9c947` - `speculative : keep MTP hidden states on backend`

- Что сделано: MTP hidden states сохраняются в backend/GPU buffers и передаются следующему MTP graph без обязательной копии через CPU.
- Зачем: убрать синхронизацию и PCIe transfer на каждом speculative шаге; первоначальный GPU-direct путь был ориентирован на один slot.

### 20. `01c44d7` - `speculative: support GPU-direct MTP with multiple slots`

- Что сделано: добавлены отдельные backend-регионы hidden states для последовательностей и graph gather для непоследовательных строк нескольких `seq_id`.
- Зачем: сохранить GPU-direct MTP при `n_slots > 1`, не смешивая состояния параллельных запросов.

### 21. `1ffe9e4` - `cuda: vectorize indexed flash attention loads`

- Что сделано: соседние F16 и Q8_0 K/V элементы загружаются группами по четыре; launch geometry для Q8_0 настроена отдельно.
- Зачем: уменьшить число инструкций и улучшить эффективность случайных чтений в `fattn-indexed`.

### 22. `469a0d9` - `cuda: tile Lightning Indexer prefill scoring`

- Что сделано: FP32 pool keys повторно используются группой prefill queries внутри tiled CUDA kernel; decode path оставлен без изменений.
- Зачем: сократить повторные чтения одних и тех же pool keys при prefill и замедление indexer по мере роста числа pools.

### 23. `bc38ed4` - `cuda: reuse temporal hints for GLM pool top-k`

- Что сделано: предыдущие top-k индексы используются как temporal hint для оценки порога и построения меньшего exact candidate set; при неподходящем hint остается полный корректный fallback.
- Зачем: использовать близость соседних decode состояний и уменьшить стоимость точного top-k без изменения результата выбора.

### 24. `fa1eb32` - `speculative: adapt MTP draft length`

- Что сделано: длина MTP draft автоматически меняется по истории acceptance отдельно для каждой sequence, с учетом `n_min`, `n_max`, `p_min` и оставшегося контекста.
- Зачем: не тратить verify compute на слишком длинный draft при низком acceptance и увеличивать draft после устойчивых полных принятий.

### 25. `de0e283` - `cuda: cache graphs by topology`

- Что сделано: CUDA Graph cache индексируется хешем topology, shapes, strides и параметров узлов; добавлен ограниченный LRU cache до 64 вариантов.
- Зачем: повторно использовать CUDA Graph для уже встречавшихся batch/shape вариантов вместо постоянного recapture при MTP и меняющемся prompt size.

### 26. `eb905c3` - `cuda: fuse GLM pool index expansion`

- Что сделано: selected-pool gather, validity expansion, tail append и compact concatenation объединены в одну CPU/CUDA операцию `KPOOL_EXPAND`; invalid элементы передаются как отрицательные indices прямо в mask-less indexed attention.
- Зачем: уменьшить количество graph nodes, временных tensors и memory traffic перед sparse attention. Для короткого decode сохранен более быстрый materialized path.

### 27. `3589bab` - `cuda: fuse GLM expert down reduction`

- Что сделано: single-token Q5_K `MUL_MAT_ID` для expert down и последующее умножение на routing weights/reduction объединены в один CUDA kernel для формы GLM-5.3-Flash `2048x4096`.
- Зачем: не материализовать восемь промежуточных expert outputs и убрать отдельные MUL/view/add kernels при decode.
- A/B: fusion можно отключить через `GGML_CUDA_MOE_DOWN_REDUCE=0`.

### 28. `4d8feda` - `cuda: extend GLM down reduction fusion to Q6_K`

- Что сделано: fused down/reduction kernel обобщен с Q5_K на Q6_K, добавлены correctness/performance tests обоих форматов.
- Зачем: включить эту оптимизацию для `UD-Q5_K_XL`, у которого expert gate/up имеют Q5_K, а `ffn_down_exps.weight` фактически хранится в Q6_K.

### 29. `549b1bb` - `cuda: generalize MoE down reduction fusion`

- Что сделано: убрана привязка fusion к единственной форме `2048x4096`; проверенный диапазон расширен до `n_ff=768..2048` и `n_embd=2048..7168` для Q5_K/Q6_K, top-8 и single-token decode.
- Зачем: применять тот же быстрый путь к совместимым Qwen-подобным, GLM и DeepSeek-подобным MoE формам. Обычная GLM-5.3 с expert down `2048x6144` также попадает в этот диапазон.

### 30. `8ccb84f` - `fix(glm5): track pool cache rebuilds per stream`

- Что сделано: глобальный флаг восстановления persistent pool-key cache заменен состоянием для каждого физического KV stream; graph build/reuse и завершение rebuild теперь работают только с фактически обработанным диапазоном streams. Добавлен регрессионный тест для двух slots с checkpoint restore.
- Зачем: восстановление checkpoint одного slot очищает pool maps всех streams. Ранее rebuild первого slot ошибочно объявлял весь cache готовым, после чего второй slot попадал в incremental path с недостаточной емкостью и завершался на `incremental pool-key update capacity is too small`.
- Проверка: CUDA Release-сборка прошла; cached/non-cached logits совпали с `max abs = 0`; новый multi-stream restore test завершился со статусом `ok`.

### 31. `d675ef7` - `cuda: extend fused MoE down reduction to low-bit quants`

- Что сделано: single-token fused expert down и weighted reduction расширены с Q5_K/Q6_K на Q3_K, IQ3_XXS и IQ4_XS. Тестовая матрица покрывает формы `768x2048`, `2048x4096`, `2048x6144` и `2048x7168` для всех пяти типов.
- Зачем: использовать специализированный decode path для `UD-Q3_K_XL`, `UD-IQ3_XXS` и `UD-IQ4_XS`, включая формы expert-down GLM-5.3-Flash и обычной GLM-5.3.
- A/B Q3_K: fused kernel быстрее unfused цепочки на 43-45% для GLM-форм на RTX 4090 и на 34-35% на RTX 3090.
- A/B IQ3_XXS: прирост составляет 35-38% на RTX 4090 и 31-33% на RTX 3090; для IQ4_XS - 27-39% и 14-18% соответственно.
- Проверка: общая CUDA correctness-регрессия завершилась со статусом `20/20`; отдельные Q3_K и IQ3_XXS проверки прошли на RTX 4090 и RTX 3090; `llama-server` собран успешно.

## Важные зависимости между изменениями

- `26cf4fe` был временным correctness fallback; полноценный мультимодальный MTP появился в `cfcdf4e`.
- `155c6a1` уточняет область применения общего MoE fusion из `40efd56`: single-token decode использует более быстрый специализированный путь.
- `7e9c947` убрал CPU round-trip для одного MTP потока, а `01c44d7` распространил GPU-direct хранение на несколько slots.
- `3589bab`, `4d8feda`, `549b1bb` и `d675ef7` являются последовательными этапами одной оптимизации: Q5_K GLM shape, затем Q6_K, другие проверенные MoE shapes и низкобитные Q3_K/IQ3_XXS/IQ4_XS форматы.
- `7588579`, `4a7b87f`, `1f0fccd` и `8ccb84f` вместе образуют persistent pool-key cache с selective invalidation, достаточной емкостью обновления и независимым rebuild для нескольких slots.

## Диагностические переключатели

- `GGML_CUDA_MOE_DOWN_REDUCE=0` отключает fused expert down reduction для A/B теста.
- `LLAMA_GLM5_INDEXED_ATTN=0` оставляет dense sparse-attention path; значение `2` принудительно включает indexed path там, где он поддерживается.
- `LLAMA_MTP_ADAPTIVE=0` отключает адаптивную длину MTP draft.
- `GGML_CUDA_GRAPH_SHAPE_CACHE=0` возвращает старую схему ключа CUDA Graph cache.

## Текущее состояние

- Последний коммит с реализацией: `d675ef7`
- Все перечисленные изменения находятся в истории ветки `glm5next-upstream-optimized`.
- Отчет сформирован по фактическому диапазону `origin/glm5next/upstream..HEAD`; документационные коммиты исключены из списка.
