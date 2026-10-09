# DeepSeek-V4.1 long-history divergence localization

This stage preserves the Stage22 general-callback.h and raw-layout.h byte for byte. It changes only diagnostic capture and analysis tools. The production sources, installed server and profiles are unchanged.

The first model process runs three cached and two fresh fixed-history replays. All use the Stage22 ratio2 prompt (6617 tokens) and Stage19 history (1024 fixed token IDs). The history belongs to another prompt; this checks numerical width dependence, not answer quality. Captures cover inputs749/750 and boundaries of all40 layers. The three cached full-logit SHA controls must retain Stage22 values before interpreting captures.

The second process captures the layer24 attention subgraph and index sources20/24 for cached widths1/4. It also runs fresh widths1/4 with capture disabled, providing full-logit controls for the fresh captures. Each process loads the model once; GPU jobs are sequential.

Reproduction: compile cache-replay.cpp using replay-build-manifest.json, then run run-replay.py and analyze.py. Run prepare-focused.py, run-focused.py --cases focused-cases.txt --output focused-output --prefix focused, then analyze-focused.py. The first analyzer reuses the established tensor reader and metrics.so from stages16/5. Python3 and the standard library suffice. The immutable32 Stage8 libraries are checked before every model process.

Full logits, binary tensors, checkpoint files, executables and full logs stay in the workspace. raw-file-sha256.json records their size and SHA. The review archive contains the sources, commands, manifests, compact tensor metadata, comparison metrics and output token metadata; it excludes model weights and large binary captures.

This callback duplicates work. No production throughput or general compatibility claim is made. Review is required before the next commit.

Recorded outcome: the first cached logit divergence at output751/query7367 is caused by width-dependent BF16 indexer projection rounding in layer24. Its exact5120x32 input is unchanged. The scaled weights differ by at most1.86e-9 and swap one selected compressed row (3607 versus1106) at top-k rank2047. Frozen replacement of only indexer W restores all scalar scores/top-k; replacing only the attention mask restores scalar FA. Reverse replacements reproduce wide. All23 variants and23 repeats pass. The projection is not fixed in this stage.

Fresh width1/4 also diverge, starting at output281/query6897; two argmax differ. Both no-capture controls are exact. The first fresh cause is not localized; do not assume it is the same as the cached cause. Full target-logit compatibility and production speedup remain open.

For frozen reproduction, run prepare-projection.py, run-projection.py, prepare-indexer.py, run-indexer.py, prepare-attention.py, run-attention.py, then analyze-isolated.py. The projection and attention runners reuse the verified stage13/stage18 binaries. GPU runs must be sequential. verify-integrity.py validates positive controls and recorded negative outcomes.
