# GLM-5.3 sampling repeatability

This stage repeats the Stage 0 raw `/completion` scenario after the upstream backports. The tested server and libraries are the immutable Stage 5 `candidate-bin` snapshot, corresponding to commit `11638b68545860e96b055798e995bb14be3d0e88`. The pending reasoning-budget Continue change is excluded. No production source files are modified by this stage.

The snapshot was built before the Stage 5 source commit. Its embedded version is `b11012-cc7bfba3b`, identifying the Stage 4 base plus the then-uncommitted Stage 5 changes. Snapshot SHA256 and the recorded Stage 5 source hashes identify the actual tested content; the embedded base commit alone does not identify it.

`run-repeatability.py` reuses the existing baseline API and memory monitor helpers. It verifies the Stage 5 snapshot hashes before each server launch and verifies the paths and hashes of loaded llama, ggml and mtmd libraries through `/proc/PID/maps`. It refuses to overwrite a run directory and stops its own server after each series, including on failure.

The model is the same seven-part `GLM-5.3-UD-IQ3_XXS` GGUF. Model sizes and modification times are compared with Stage 0. The original request objects are reused byte-for-byte at the JSON value level: 96 facts, 1656 prompt tokens, 128 generated tokens, seed 1234, cache disabled, top-k 40, top-p 0.95 and min-p 0.05. Sampling uses temperature 0.8; greedy uses zero. The server uses context 8192, batch 2048, ubatch 512, one slot, F16 target/draft KV, 12 CPU threads, layer split and automatic fit with a 3072 MiB reserve. MTP uses the embedded NextN and maximum draft length 3.

There are four sequential launches. Each of the two MTP launches performs five identical sampling requests followed by five identical greedy requests. Each of the two launches without MTP performs five identical sampling requests. All 30 request/response pairs retain full generated token IDs, text, server timings, and draft/accepted counters. Every response is checked for a complete token array and zero cached prompt tokens.

The six GPUs are fixed by UUID, using the same order as the post-reboot Stage 4 and Stage 5 runs. Stage 0 used hardware that was later replaced, so these results alone cannot establish which source change or hardware change caused a difference from the original baseline. `runs/gpus.csv` records the current devices and driver. Each manifest records the final assignment of every layer, actual loaded libraries, model buffers, KV buffers and compute buffers. Full fit attempts and diagnostic output remain in the local server logs.

Run from the workspace with the existing Stage 0, Stage 4 and Stage 5 artifacts available:

```bash
python3 llama_upstream_review/stage6/run-repeatability.py \
  --output /home/sergei/_my_sync/llama_upstream_review/stage6/repeat-runs
python3 llama_upstream_review/stage6/analyze.py \
  --runs /home/sergei/_my_sync/llama_upstream_review/stage6/repeat-runs
```

`analyze.py` checks all pairs within each launch and across the two launches of each mode. It records token and text equality, unique sequences, draft acceptance, and the first different token with zero-based and one-based indices. It also checks exact request equality to Stage 0, model identity by size/mtime, prompt and output token counts, and placement consistency between restarts. A successful inference status does not mean repeatability passed; the `all_token_ids_identical` and `all_text_identical` fields express that result.

The Git copy contains the runners, summary, manifests, props and request/response JSON. The matching binaries, raw server logs and one-second memory samples stay in `/home/sergei/_my_sync/llama_upstream_review`. Copy the sibling baseline helper and matching historical artifacts when reproducing elsewhere; binary snapshots are machine-specific.

This is a repeatability check for one prompt, one seed, short context and one slot. It does not establish determinism for other prompts, other sampling parameters, parallel requests or arbitrary context lengths. It does not compare answer quality or claim a model speedup. A persistent difference is reported for separate diagnosis, as required by the plan.
