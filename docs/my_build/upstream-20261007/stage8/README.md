# Stage 8 artifacts

Q4_K comparison uses the unchanged Stage 5 CUDA snapshot. FP32 controls use two separate snapshots: `candidate-bin` for the CUDA precision fix and `final-bin` for the excluded experiment with explicit V4.1 HC precision. The working source retains only the CUDA precision fix; the HC flag was removed after the HTTP control still diverged. Both Stage 8 snapshots include the independent pending reasoning-budget Continue fix.

Large GGUF files, raw tensors, float logits, binaries and complete logs remain in the local workspace at `/home/sergei/_my_sync/llama_upstream_review/stage8/`. They are excluded from the repository archive. Commands and model paths refer to that workspace and the existing local models. No model or installed server was replaced.

Reproduction order:

1. `q4k-quantize-command.json` and `q4k-tensor-types.txt`: create the selective Q4_K copy with the Stage 5 libraries. `verify-q4-copy.py` verifies all 69 preserved tensors.
2. `run-q4-comparison.py`: run Q4_K, then MXFP4 with the Stage 5 runner. Each configuration issues 18 completions. `analyze-q4.py` checks repeats, equality across formats and the timing summary.
3. `capture-target.cpp` / `run-capture.py`: diagnostic intermediate tensor capture. Its wide-batch logits differ from the original uncaptured path, so this capture is excluded as a strict production reference.
4. `extract-router.py` and `router-replay.cpp`: replay the actual BF16 router weights and one fixed F32 input against a CPU double reference. `router-replay-before.jsonl` and `router-replay-after.jsonl` record the results.
5. `run-precision-replay.py`: server-matched teacher forcing with the CUDA-only fix. `run-hc-precision.py`: separate HC precision prototype; it mutates precision through a diagnostic callback.
6. `run-final-replay.py`: experimental model graph with explicit HC precision and no callback. `final-vs-hc-prototype.json` checks all three float dumps against the prototype. `run-final-server.py`: actual greedy server controls with DSpark off/on.

Use fresh output directories and run GPU stages sequentially. The final server controls use the original MXFP4 sidecar. Quantized Q4_K controls are not a comparison with a separately converted high-precision checkpoint. Teacher forcing uses the original Stage 7 native prefix for all widths; it is not a native-answer reference for the modified build.

The report is `../STAGE8_REVIEW.md`. Remaining numerical and throughput limits are stated there.
