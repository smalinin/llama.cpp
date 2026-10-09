# Stage 30: GLM5NEXT RAM restore

Base commit: 9df5a4a40 (Stage29 review/results), retaining Stage27 tail rollback and Stage28 unified multi-slot fixes. GPU processes run sequentially. Installed server, build-glm53 binaries and presets remain untouched.

The Stage29 native RAM control is reproduced with three ALPHA/BETA/GAMMA prompts, greedy sampling, 64 outputs, unified KV, three slots, F16 KV, context 8192, batch 2048, ubatch 64, RAM cache 2048 MiB, cache-idle-slots enabled and the same six GPU UUIDs. The existing Stage29, Stage2 and baseline drivers supply model paths and lifecycle handling. Pool-cache is disabled for native diagnostics; speculative final controls enable it.

## Diagnosis

- `ram-replay.cpp` and `run-cpu.py`: forced-token replay on the existing tiny GLM5NEXT fixture. The initial eager-clear schedule is exact. `ram-occupied-replay.cpp` and `ram-state-replay.cpp` reproduce restore-before-idle-clear; canonical KV bytes and recurrent checkpoints remain exact, with tiny logit differences and unchanged argmax. `run-gpu.py` applies the same check on one GPU.
- `make-trace.py`, `build.py`, `run-server.py --label trace-native --build trace --resident`: metadata-only instrumentation verifies main/indexer KV layout agreement. The same three prompts are also repeated in a resident slot, where all token/probability comparisons are exact.
- `make-roundtrip.py`, `build-context.py`: save every restored host state immediately back to bytes. All 15 full/partial roundtrips are exact; all 12 responses match the baseline.
- `make-capture.py`: broad first-four-layer capture. It changes some baseline responses and is rejected as proof of the original full execution. Generic tensor names can repeat and overwrite the same dump; only unique names can be used for that exploratory capture.
- `make-fa-capture.py`, `build-context.py --label fa-capture --source .../fa-capture-llama-context.cpp`: capture only the first FlashAttention operation on the four-token ALPHA suffix. Set `STAGE30_CAPTURE` to a new absolute dump directory before `run-server.py`. All 12 responses and returned probabilities remain identical to the baseline.
- `prepare-fa.py` matches visible F16 KV rows by their complete 1024-byte contents. Q, the 435 visible KV rows and logical masks agree; physical order/extent differ (512 versus 1024 cells). `fa-replay.cpp` independently reproduces the captured CUDA outputs. Restoring the reference layout also restores the exact reference output. This localizes the observed numerical difference to FlashAttention with a different physical layout; it does not establish an individual CUDA instruction defect.
- `analyze-diagnostics.py` verifies the baseline controls and records the diagnostic comparisons. `analyze-capture.py` records exploratory tensors only; use the narrow FA controls for the accepted conclusion.

## Candidate and bounded validation

`make-preclear.py` prepares the minimal server change: before loading a RAM prompt, save and clear other idle slots when idle-slot caching and unified KV are enabled. Keep the selected slot and every processing slot. The existing post-launch idle handling stays in place for the other scheduling paths. No model arithmetic or serialization format changes.

`build-server.py --label preclear --source .../preclear-server-context.cpp` compiles only server-context.cpp and links an isolated server implementation. It copies the immutable Stage28 libraries, hashes build inputs and does not rebuild or replace the model/CUDA libraries.

`run-final-server.py` executes the original 12 RAM requests, optional identical resident repetitions (`--resident`), and two waves of three simultaneous requests (`--parallel`). Final modes are native (pool-cache off), MTP and DFlash (pool-cache on). HTTP status alone is not a reproducibility pass; token and returned-probability comparisons are recorded separately. Speculative probability fields have the coverage limitations documented in Stage29: empty top_logprobs rows are not full numerical observations.

Use new labels/output directories for reruns; existing captures and binary snapshots are immutable. Raw logs, binaries, states and tensor dumps stay in the workspace. The review archive retains scripts, patches, compact responses, summaries, manifests and SHA indexes. The fix targets the observed idle unified RAM lifecycle; arbitrary concurrent layouts and file-session restores are outside this validation.
