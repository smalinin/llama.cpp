# Stage 33: DeepSeek-V4.1 speculative arithmetic

Base: `my_build` at `135c2110f`. The user approved the change and closed the DSpark item. Installed binaries and profiles are unchanged.

The production graph now uses the same arithmetic for short single-sequence verification and single-token decode. The predicate requires all token outputs and a width no larger than the recurrent rollback window plus its anchor. Floating projections use FP32 and separate query columns, including grouped output projections. Routed MoE uses the existing single-query graph for each query. Raw attention gathers each query's own logical SWA window; both compression-plan slots use per-query padding, and graph reuse checks those extents. Large prefill batches and other model architectures retain their regular batch paths. No diagnostic evaluation callback or duplicate output replacement is used.

`source.patch` contains the source changes. The installed server and profiles are untouched.

## Reproduction

The recorded commands use `/home/sergei/_my_sync/llama_upstream_review/stage33/` for isolated binaries and large local artifacts, and `/home/sergei/Github/llama.cpp` for sources. Model paths are in server manifests. Do not overwrite a recorded output directory; choose a fresh label for new server runs.

- `build-final.py`: reuse the Stage32 immutable executable/GGML snapshot and rebuild libllama with the original Release compile/link commands. Private-header dependents are rebuilt, and the manifest records source, object and library hashes. No original build object or installed library is replaced.
- `run-server.py LABEL BIN --graphs [--draft 1|3] --repeat-short`: start an isolated localhost server, verify its loaded library paths/hashes, and run four short greedy requests twice plus a 6617-token fresh/cache pair with 1024 forced output tokens. The second short pass measures warmed graph behavior. `ignore_eos` is enabled to keep lengths equal. Draft acceptance and real cache hits are checked. The harness always stops its own server.
- `replay.cpp`: public C API replay with no evaluation callback, comparing every vocabulary logit at widths 1/2/3/4 and after partial tail rollback. Native context uses n_rs_seq=0; speculative contexts use 1/2/3. Fixed input history separates target arithmetic from draft quality and sampling. The real history covers positions 6617 through 7640. `run-final-replay.py` records final-binary CPU/GPU-independent inputs and executes GPU fixtures followed by the real model.
- `make-fixture.py`: adapt the existing test-llama-archs fixture outside tests/, with 64-wide attention heads, 32-wide indexer heads and compression ratios 1/2. Candidate-mask reuse starts in the final layer, avoiding an invalid cross-ratio mask in this synthetic fixture. F16/Q8 KV are tested on CPU/CUDA.
- `run-final-regression.py`: existing DeepSeek state/save/load/rollback, batch allocation, tail rollback, GLM sparse and GLM restored-layout checks, using the final shared library.
- `analyze.py`: verify saved responses, token counts, cache hits, actual speculative work, source/library identity, replay results and regression status. Large logs and generated models remain local; the compact review archive contains results, scripts, manifests and hashes.

## Interpretation

Token equality applies to the checked greedy requests and full-logit equality to the checked fixed histories. It does not establish identical sampling across arbitrary hardware, batches or concurrent sequence layouts. The real model uses F16 KV, context8192, one slot and six GPUs; Q8 is checked on the small fixture. The implementation has no hard-coded dependency on the tested prompts or token IDs.

Correct arithmetic and useful acceleration are separate criteria. Report cold and warmed throughput separately: speculative graph capture has a visible first-pass cost. The final review records the measured decision for this target, MXFP4 sidecar and hardware; it does not claim a universal impossibility of accelerating DSpark.

Intermediate failures are retained locally: the first production candidate missed the second compression-plan slot; the first mixed fixture reused an incompatible candidate mask across ratios. Neither is counted as a passing final result. The grouped floating projection refinement removed the small CUDA differences in the original fixture.
