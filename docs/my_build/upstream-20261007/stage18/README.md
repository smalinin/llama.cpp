# DeepSeek-V4.1 raw and compressed KV padding diagnosis

The Stage17 replay first diverged when a batch included absolute position256. This stage captures input231/232 of the 23-token explain prompt, preserves all four Stage17 logit hashes, and isolates padding effects in layer0 and layer20 flash attention.

`diagnostic-callback.h` is the unchanged Stage17 control. `candidate-callback.h` is v1, which restores the first raw-cache boundary only. `v2-callback.h` also restores the compressed-cache extent for ratio1 layers. Both candidates remain diagnostic callbacks and only handle the first boundary at absolute position256. Removed rows must have negative-infinite masks. No production kernel, installed server or model profile is changed.

Scripts use immutable `../stage8/candidate-bin` libraries and the recorded GPU UUID order. Run GPU jobs sequentially. Output directories must not already exist; preserve completed outputs before a rerun.

1. `python3 prepare-capture.py`; `python3 run-capture.py`; `python3 analyze-capture.py`.
2. `python3 prepare-attention.py`; `python3 run-attention.py`; `python3 analyze-attention.py`.
3. `python3 run-profile.py`; `python3 analyze-profile.py`.
4. `python3 prepare-candidate.py`; `python3 run-candidate.py`; `python3 analyze-candidate.py`.
5. Compile and run `candidate-capture.cpp` with the commands in `candidate-capture-build-manifest.json` and `candidate-capture-run-manifest.json`; `python3 analyze-candidate-capture.py`.
6. `python3 prepare-v2.py`; `python3 run-v2.py`; `python3 analyze-v2.py`.
7. `python3 prepare-layer20.py`; `python3 run-layer20.py`; `python3 analyze-layer20.py`.
8. `python3 run-layer20-profile.py`; `python3 analyze-layer20-profile.py`.
9. `python3 prepare-build-v2.py`; `python3 run-free-generation.py`; `python3 analyze-free.py`.

`v2-replay` loads the model once for two v1 capture controls, four explain256 v2 replays and four baseline95 v2 replays. `v1-callback.h` only renames the v1 helper namespace so both versions can be included in this executable. The two v1 SHA controls verify this combination.

The v1 experimental server was compiled but never run. Only `experiment-v2-bin` is used by this stage's HTTP runner. The sole changed snapshot library is `libllama-server-impl.so`; callbacks remain disabled in the draft model and prefill. Precision is restored and cached down inputs are cleared after each decode.

Raw tensor dumps, full FP32 logits, model/server logs, profiles, binaries and build objects stay in the workspace. Review archives contain sources, manifests, compact results, alignments, tensor metadata and hashes, without model weights or large raw dumps. Diagnostic HTTP timings include duplicated operations, CPU transfers, allocation and synchronization, and are not production speed measurements.
