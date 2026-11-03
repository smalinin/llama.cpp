# Паспорт модели: <имя>

Дата / runtime commit / dirty patch hash:
Назначение и целевой режим (prefill, TG, TTFT, parallel, multimodal):

| Поле | Значение и источник |
| --- | --- |
| GGUF/weights identity | Paths/size/hashes или immutable source revision |
| Graph architecture | Layers, hidden/head dims, dense/MoE, expert shapes/picks |
| Actual tensor precision | По projection/layer, combined/separate gate/up, embeddings/Engram |
| KV topology | Source/sharing/compression/SWA/sinks, physical versus live capacity |
| Consumers | Output rows, draft/MTP taps, indexer, embeddings/media |
| Draft | Trained architecture, heads, extract IDs, widths, masks, bridge |
| Backend | GPU architecture/UUID/PCI order, CPU, driver/toolchain, build flags |
| Placement | Actual model/tensors/cache/KV/workspace by device, host spill |
| Launch | Expanded argv/environment, batch/ubatch/ctx/parallel, sampling |
| Workload | Frozen IDs/media/template, corpora, output limits, reuse policy |
| Baseline | PP/TG/TTFT/load, measured memory, acceptance denominator |

## Applicability decision

Для каждого выбранного ID из CATALOG: поддержан / неприменим / требует проверки; указать причину и fallback. Константы из DeepSeek брать только после собственного dependency/shape proof.

## Validation contract

Reference и ожидаемая exactness/error policy, NLL/task metrics, lifecycle cases и согласованный memory floor. Непроверенные режимы перечислить явно.

## Results and next action

Ссылки на локальные experiment records/evidence, завершённые и открытые gates. Отделить измеренные результаты от текущего default.
