# DeepSeek-V4.1 diagnostic indexer projection fix

Stage23 isolated the cached query7367 logit divergence to BF16 indexer projection rounding. A width-dependent difference of about1e-9 after scaling changes one selected top-k row. This stage extends the established scalar floating-MUL_MAT intervention to indexer projections during generation.

The new opt-in mode suffix is indexer. The selector requires a floating weight type, 2D operands, widths2..4 and a weight name beginning with blk. and ending with .indexer.proj.weight. The intermediate MUL_MAT node has no stable model label, so the weight name identifies the operation. The existing per-column views, GGML_PREC_F32 matmul and concatenation are reused. Width1 and prefill retain the previous path. Raw layout, attention extents, compressor/router/HC and routed-FFN policies are unchanged.

Indexer counters record the number of replaced query columns by original weight name. The model analyzer requires the exact eight sources2/8/14/20/24/28/32/36 and the expected count for every completed case. A final scalar tail is excluded from replacement counts.

Reproduction: run prepare.py, run-projection.py, analyze-projection.py, run-replay.py and analyze-model.py, record gpu-after.csv with nvidia-smi, then run verify-integrity.py. prepare.py derives the small header patch and harnesses from the immutable Stage23/Stage22/Stage13 sources and records compiler commands. The projection check uses the preserved Stage23 frozen weights and activations. All GPU jobs must run sequentially; Stage8 snapshot hashes are verified before each run. Output directories must be absent before each replay; preserve the recorded runs when repeating elsewhere.

The bounded model matrix has12 cases: four previous baseline/sparse/history SHA controls and two long histories (cached/fresh) at widths1/2/3/4. The long prompt is the Stage22 ratio2 window with6617 tokens; the1024 fixed IDs come from Stage19 on another prompt. This is an arithmetic comparison, not free-answer quality. Fresh/cached scalar controls must preserve Stage23/Stage22 SHA respectively; corrected wide outputs are compared with those scalar histories, not with old divergent wide hashes.

The callback still duplicates GPU work and is diagnostic. No production source, installed server, profiles or immutable libraries are changed. Free DSpark generation after this new fix and production CUDA throughput require the next review stage.

Large logits, checkpoints, logs and binaries stay in the workspace; raw-file-sha256.json records their hashes. The repository review archive includes sources, manifests, compact metadata, comparison summaries and indexes without weights or large binary dumps.

Recorded outcome: all12 model cases completed. All6 previous full-logit SHA controls are preserved. The cached and fresh1024-row histories are bit-identical across widths1/2/3/4: six wide/scalar comparisons, max/RMS0/0 and no argmax differences. Cached SHA is6b0d5aef9cbd7456149d97fa5b31e6ac4f1d56e3d41823758bb77ad444770e30; fresh SHA is2f04b5807f7dbe708a51c790c570d34a7b07d495946b88490c49352769ecde57. The previous fresh query6897 output divergence is absent with this intervention; its first operation was not separately captured.

All11 frozen projection variants and11 repeats are exact. The actual wide batch target column matches the preserved scalar projection. Indexer selection counters match every case. Physical traces match scalar for all six long wide runs, including restored raw extents256/512/768. All32 snapshot hashes remain unchanged. No new HTTP or free DSpark requests were run. This stage is prepared for review before commit.
