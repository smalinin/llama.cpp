# Stage 12 review artifacts

Diagnostic scalar routed up/gate/SwiGLU with native batched down. Same-input hidden becomes bit-identical on Ada/Ampere and local FFN amplification is removed. Whole-model greedy compatibility still fails: adding the intervention to FP32 floating matmuls and scalar attention changes which width diverges at index19 and increases final logit max/RMS versus that previous control.

66 isolated FFN variants were computed twice, including the profile controls; 12 local benchmark cases use 10 warmups and 5 groups of 100 calls. All 12 full-model replays and 8 logit SHA controls passed. See ../STAGE12_REVIEW.md, integrity.json and the numeric summaries. General DSpark compatibility and performance remain open; no production code or installed server change.

Reproduction requires original model files, Stage 11 inputs/weights index, Stage 7 prompt, Stage 9 forced history and Stage 8 immutable candidate libraries from the local workspace. Build commands are in build-commands.json. Run run-controls.py for ada and ampere, and ada --profile; the profile binary omits benchmark iterations. Export its Nsight report to controls-ada-profile.sqlite, then run analyze-controls.py and analyze-profile.py. run-target-ffn.py performs the whole-model controls; analyze-target.py and verify-integrity.py validate results. Each run requires a fresh output directory.

Actual commands and source/binary/input hashes are in manifests. Runtime library paths/hashes and model allocation lines are preserved. Curated short logs use .txt; the original .log files remain in the workspace. Source fragment scalar-upgate-fragment.cpp was only a preparation helper and is not needed to build target-ffn-controls.cpp.

Large tensor/logit dumps, binaries, weights and traces remain in /home/sergei/_my_sync/llama_upstream_review/stage12/. raw-artifact-sha256.json identifies those local data; target-runtime-summary.json identifies the complete model log. Prepared for review before commit.
