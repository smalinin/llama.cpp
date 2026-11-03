# Локальные доказательства

Этот каталог входит в runtime Git-ветку. Выбранные исходные reports сохранены дословно в `reports/*.txt`; исходные Markdown links в этих snapshots относятся к прежнему notebook tree. Рабочие локальные точки входа находятся ниже. [provenance.json](provenance.json) содержит исходный commit/path и hashes, поэтому при необходимости можно найти полный исторический архив. Для чтения базы доступ к нему не требуется.

Исторические reports сохраняют статусы/defaults на момент своих опытов. Актуальное состояние на снимке см. в [каталоге](../CATALOG.md), [паспорте](../models/deepseek41-1017.md) и [runtime README](../../../readme_dsv41.md).

## E01

M2 + BF16 DSpark, final gates: [исходный report](reports/E01.txt). Источник `validation/moe-cache/1017-dspark-prefill-skip-20261103/REPORT.md`.

## E02

CPU MXFP8 LUT: [исходный report](reports/E02.txt). Источник `validation/moe-cache/1017-mxfp8-lut-20261101/REPORT.md`.

## E03

Q3 MMQ32 + DSpark: [исходный report](reports/E03.txt). Источник `validation/moe-cache/1017-q3-mmq-dspark-20261031/REPORT.md`.

## E04

Q8 paired fit and native gates: [исходный report](reports/E04.txt). Источник `validation/moe-cache/1017-q8-paired-fit512-20261101/REPORT.md`.

## E05

Producer-local feature retention: [исходный report](reports/E05.txt). Источник `validation/moe-cache/1017-feature-local-20261102/REPORT.md`.

## E06

MoE startup calibration: [исходный report](reports/E06.txt). Источник `validation/moe-cache/1017-prefill-auto-calibration-20261030/REPORT.md`.

## E07

Rejected serial FP16/GEMM: [исходный report](reports/E07.txt). Источник `validation/moe-cache/1017-prefill-fp16-screen-20261031/REPORT.md`.

## E08

Rejected FFN graph regions: [исходный report](reports/E08.txt). Источник `validation/moe-cache/1017-prefill-regions-20261031/REPORT.md`.

## E09

Central GPU bridge transfer screen: [исходный report](reports/E09.txt). Источник `validation/moe-cache/1017-gpu-feature-transfer-20261102/REPORT.md`.

## E10

Async ingress screen: [исходный report](reports/E10.txt). Источник `validation/moe-cache/1017-feature-ingress-async-20261102/REPORT.md`.

## E11

Engram lookup ownership screen: [исходный report](reports/E11.txt). Источник `validation/moe-cache/1017-engram-owned-lookup-20261101/REPORT.md`.

## E12

M2 CPU tail-state proof: [исходный report](reports/E12.txt). Источник `validation/moe-cache/1017-dspark-tail-screen-20261103/REPORT.md`.

## E13

Q8 dual-buffer verification: [исходный report](reports/E13.txt). Источник `validation/moe-cache/1017-q8-dual-buffer-20261102/REPORT.md`.

## E14

F16/Q8 memory and speed tradeoff: [исходный report](reports/E14.txt). Источник `validation/moe-cache/1017-q8-f16-paired-ab-20261102/REPORT.md`.

## Reproduction bundles

- [M2 + DSpark: протокол](reproduction/dspark-prefill/PROTOCOL.md), [server summary](reproduction/dspark-prefill/summary.json), [raw requests/argv/env/hashes](reproduction/dspark-prefill/bench-results.json), [long gate](reproduction/dspark-prefill/long-summary.json), [lifetime guard](reproduction/dspark-prefill/lifetime-summary.json), [artifact hashes](reproduction/dspark-prefill/SHA256.json).
- [CPU tail-state proof](reproduction/dspark-tail-proof/results.json), [cone cases](reproduction/dspark-tail-proof/cone-results.json), [fixture source](reproduction/dspark-tail-proof/cache.cpp), [artifact hashes](reproduction/dspark-tail-proof/SHA256.json).
- [Исходный M2 design](reports/M2-design.txt).

Bundles содержат frozen token IDs, helper sources, manifests, patches и raw logs. Веса и executables не включены. Абсолютные пути в исторических runners/argv нужно явно переназначить; запуск из новой папки без remapping не обещается. Например, старые M2 helpers трактовали unset как OFF, а в6d038260e unset включает skip; при переносе проверок нужно явно задать arms и актуализировать ожидания. Helpers используют internal ABI: собирать с headers и библиотеками соответствующего runtime. Профилирующие fault observers применялись только в validation, не в timed A/B.

`bench.patch`/bench hashes фиксируют измеренную сборку. `final.patch`/final-build фиксируют последующий stale-range guard до default ON. Runtime commit6d038260e дополнительно включает default ON. Эти фазы не выдаются за один бинарник и не переписываются.

Контроль целостности всего каталога: [SHA256.json](SHA256.json). Он покрывает snapshots, bundles, provenance и этот индекс; внутренние manifests сохранены без изменений. Для проверки средствами Python из каталога evidence:

```bash
python3 - <<'PY_CHECK'
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('SHA256.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Evidence checksums OK')
PY_CHECK
```
