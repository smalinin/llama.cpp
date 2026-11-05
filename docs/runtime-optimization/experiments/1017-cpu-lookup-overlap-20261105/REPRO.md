# Evidence and reproduction

This is a single-mechanism scheduler experiment based on `my_maya aa33d0490`. See [REPORT.md](REPORT.md) for the result and scope. Everything measured is from the new host; old-host percentages are not reference data.

- `artifacts/final-build.json`: exact candidate compile/link argv and source/library fingerprints. The production library is unchanged. `artifacts/build.py` builds an isolated scheduler library using the existing compatible CMake flags/objects.
- `artifacts/native.py` and `native-results.json`: native timing controller, expanded environment/argv, unchanged production library hashes, frozen model metadata, loaded candidate maps, complete responses/counters, memory and exits.
- `artifacts/scheduler.cpp`, `scheduler-results.json`, `unit-manifest.json`: external CUDA/CPU scheduler fixture and 42 cases against the same private library.
- `artifacts/precision.cpp`, `precision.py`, `precision-inputs.json`, `fault.cpp`: raw-logit/tap/KV/state/restore/rollback/NLL and failure/retry gate. Two fresh OFF/ON processes, 12 groups each, two fault recoveries each. Results and logs are in `precision-0/` and `precision-1/`.
- `artifacts/counts.cpp`, `run-counts.py`, `counts.jsonl`: small baseline event-sync/CPU graph counter scopes. The recorded binary's disassembly offsets are not portable to another build.
- `artifacts/summarize.py`: derive the native means/deltas and counter totals; it reads the recorded manifests and writes `summary.json`.
- `artifacts/review.py`, `review-checks.json`: final source/library/model-identity, precision/memory/error-span and link/whitespace checks. `review-notes.json` preserves metadata-review corrections.

The controllers are historical single-use snapshots with absolute paths, ports and explicit UUID mapping. They refuse to overwrite recorded results. Preserve these artifacts; copy/remap the controller paths and output root for a new run. The native controller's admission check originally expected the 36-case fixture available before measurement; a fresh run should require the final 42-case manifest. Keep the recorded snapshot unchanged.

Rebuild external fixtures against the same internal headers and shared libraries as the runtime. The precision fixture links llama-common/llama/ggml and OpenSSL crypto; the scheduler fixture links ggml CPU/CUDA. The small counter/fault libraries interpose the recorded ABI and need rebuilding for another binary. Do not use old absolute callsite offsets on another build.

Before any repeat, verify free GPUs, current UUID/PCIe order, CPU flags, input IDs, model tensor/KV types, placement, sampling and actual memory floor. Choose a free local port. Start timing arms sequentially with only `GGML_SCHED_CPU_LOOKUP_EARLY=0/1/0` changing, and no interposer, CUPTI or callback in native timing. Raw precision is separate and disables backend sampling. Do not rebuild or replace a library while a process is loading/using it.

Binary/object files and private candidate directories are deliberately ignored. `candidate-final` was a temporary redundant-guard audit build and is not the final candidate. Failed negative-fixture expectations, the controlled pre-READY loader cancellation, original incomplete supervisor manifests and all measured source snapshots remain available. They are not counted as passed runtime tests.

## Current default

The original measured controllers/fixture describe the initial opt-in default OFF. The later `artifacts/promotion/` fixture expects unset to match ON; use that version when reproducing the current default. Its private and installed-module manifests each record 42 passing cases. `default-only.patch` separates the constructor-default change from the original scheduler implementation. Production library/build metadata and controlled cancellation of the unnecessary full CUDA rebuild are recorded separately; the old fingerprints remain historical.
