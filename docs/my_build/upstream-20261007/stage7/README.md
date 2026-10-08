# DeepSeek-V4.1 DSpark diagnosis

This stage uses the immutable Stage 5 CUDA Release server and libraries, corresponding to source commit `11638b68545860e96b055798e995bb14be3d0e88`. The embedded pre-commit build version is `b11012-cc7bfba3b`; snapshot hashes identify the actual tested content. The separate pending reasoning-budget Continue fix is excluded. The installed user server and its configuration are not changed.

`inspect-models.py` reads GGUF metadata and tensor headers without reading model weight payloads. It checks tokenizer array hashes, dimensions, target extraction layers, DSpark semantics and confidence/Markov tensors. Metadata does not contain an exact source checkpoint revision. Model names and structural compatibility do not establish a training revision match. The available V4-0731 drafts have a different hidden size and target layer list, so they cannot serve as an alternative precision control for this V4.1 target.

`run-dspark.py` runs six separate server configurations: native, draft lengths 1/2/3 with confidence disabled, and length 3 with thresholds 0.3/0.6. HTTP overrides for speculative parameters are disabled in this source tree; CLI restarts are necessary. The target model placement is fixed with `fit=off`, layer split and tensor split `1,1,1,1,1,0.4`. GPU UUID order is retained in every manifest. Context is 8192, batch 2048, ubatch 512, one slot, F16 target/draft KV and 12 CPU threads. Snapshot and all loaded llama/ggml/mtmd library hashes are checked at each launch.

Each configuration runs two 128-token baseline diagnostics, four warmups, and three rotated measurement rounds for four prompts (18 requests). The original raw baseline prompt contains 1655 tokens for this model. Three additional prompts request HTML/SVG, Python interval merging and an explanation of database indexes. They are rendered through `/apply-template` with thinking disabled, and their rendered forms are saved. Generation is greedy with seed 1234, cache disabled, returned token IDs and an output limit of 256. Natural EOS is respected. Output equality and lengths are reported separately from throughput; different answers are not an equivalent-work speedup.

The benchmark runs without `LLAMA_DSPARK_PROFILE` or Nsight. One profile run uses the existing stage timers and a separate Nsight Systems CUDA/NVTX trace. Capture is gated around `baseline-diagnostic-2` after the initial request; diagnostic 3 runs after capture stops. Node-level CUDA graph tracing and synchronization affect timing, so profiled times are diagnostic and do not replace benchmark throughput. The fixed verification width zero control pays for the draft but sends no draft tokens to target verification; it checks whether enabling hidden extraction/injection alone changes the output.

`target-replay.cpp` uses the same saved llama/ggml libraries, loads the target once and teacher forces the first 64 native output tokens. New contexts compare decode widths 1/2/4, extraction of layers 37/38/39 and recurrent rollback reserve 0/1/3. There is no draft model and no rollback during this replay. It uses the server's `swa_full=false` and checkpoint prefill boundaries 1139/512/4. The 65 full logit rows per variant and the original forced tokens distinguish differences in target computation from draft token selection. Contexts reserve four output rows, with an additional one-row native control. Native single-token argmax must first be checked against the server reference. Two earlier replay attempts with unmatched defaults are excluded from conclusions about the server; their raw artifacts remain in the workspace.

Run from the local workspace, with the existing Stage 0, Stage 4 and Stage 5 artifacts available. Use a new output directory when repeating a run:

```bash
python3 llama_upstream_review/stage7/inspect-models.py
python3 llama_upstream_review/stage7/run-dspark.py
python3 llama_upstream_review/stage7/analyze.py benchmark llama_upstream_review/stage7/runs
```

Compile the replay against the snapshot:

```bash
c++ -std=c++17 -O2 \
  -I /home/sergei/Github/llama.cpp/include \
  -I /home/sergei/Github/llama.cpp/src \
  -I /home/sergei/Github/llama.cpp/ggml/include \
  llama_upstream_review/stage7/target-replay.cpp \
  -L /home/sergei/_my_sync/llama_upstream_review/stage5/candidate-bin \
  -Wl,-rpath,/home/sergei/_my_sync/llama_upstream_review/stage5/candidate-bin \
  -lllama -lggml -lggml-base -lggml-cuda -lggml-cpu \
  -o llama_upstream_review/stage7/target-replay-server-matched
python3 llama_upstream_review/stage7/run-controls.py
```

`run-controls.py` requires all six benchmark servers to have exited successfully. It runs the replay, Nsight and width zero control sequentially and stops on failure. It needs `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys`. The existing baseline helper, immutable binaries and historical artifacts are required when reproducing elsewhere; these scripts are specific to the local GPU/model configuration. Replay and control output directories must not exist; use a fresh artifact tree or adjust paths for another run. `analyze-nsys.py` reads the SQLite export and correlates copy activities with their initiating CUDA APIs.

The Git copy retains scripts, summaries, model metadata, manifests, profile events and request/response JSON. Raw server logs, memory samples, binaries, full float logits and Nsight reports remain in the workspace. All output token comparisons use zero-based indices unless stated otherwise. Successful inference status alone does not establish output equality, quality or a performance improvement.
