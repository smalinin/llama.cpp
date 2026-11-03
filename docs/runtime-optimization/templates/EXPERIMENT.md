# Эксперимент: <ID и механизм>

Дата / модель / runtime base / candidate patch / библиотечные hashes:
Статус: measured candidate / shipped opt-in / shipped default / rejected / inconclusive.

## Проблема и гипотеза

Measured bottleneck, ожидаемая причина и изменяемый механизм. Поддержанные shapes/backend/consumers, eligibility и fallback. Где изменяется precision/reduction/layout/lifetime. Как подтвердить, что нужная ветвь действительно выполнялась.

## Frozen protocol

Expanded argv/environment, UUID/PCI mapping, model/draft identity, KV, actual placement, input IDs/media hashes и output/stop policy. Sampling/seed, K/admission и определения PP/TG/acceptance. Warmup/repeats/order, cache reuse controls, profilerOFF timing. Memory floor, watcher, cleanup и правило остановки.

## Correctness

| Gate | Reference | Cases / result | Evidence |
| --- | --- | --- | --- |
| Operator / negative controls | | | |
| Logits / finite / NLL text-code | | | |
| Target/draft states and taps | | | |
| Append/restore/reuse/rollback | | | |
| Failure/recovery and lifetime | | | |
| Metadata/fit/real/peak memory | | | |

Использовать passed/failed/inconclusive/not run; не заменять пропущенный gate результатом другого уровня. При изменении arithmetic заранее записать допустимую ошибку и quality policy.

## Native speed

| Workload | Baseline PP/TG/wall | Candidate PP/TG/wall | Median/range/repeats | Actual memory |
| --- | --- | --- | --- | --- |

IDs/counters, branch activation и placement совпадения/расхождения; источники вариации. Не суммировать независимые проценты и не переносить operator gain на whole model.

## Decision and reproduction

Решение и условия повторного рассмотрения, scope неподдержанных режимов, source identity измеренной и promoted сборок. Локальные raw metrics, inputs, helper source, protocol и SHA256 manifest; locator больших внешних artifacts. Список собственных процессов и подтверждение завершения. Обновлённые CATALOG/LESSONS/model passport.
