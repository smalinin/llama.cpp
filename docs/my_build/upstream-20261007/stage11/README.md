# Stage 11 review artifacts

Isolated layer-2 FFN diagnosis with original DeepSeek-V4.1 weights. Six full outputs match Stage 10 model capture exactly across Ada/Ampere and widths 1/2/4. Fixed-input routed hidden crosses two Q8_1 rounding boundaries; the resulting down projection explains the local FFN amplification. General model/DSpark greedy compatibility remains open.

44 full FFN variants, 18 frozen-down variants and 6 real CUDA Q8_1 captures completed successfully, each computed twice. Six additional local FFN benchmark variants estimate the cost of full scalarization; they do not measure model TPS or the cost of changing only routed up/gate. See ../STAGE11_REVIEW.md, integrity.json and the numeric summaries.

Reproduction uses the original model path, prior Stage 7/10 captures and immutable Stage 8 candidate libraries in the local workspace. Exact GPU UUIDs, commands, source/binary/input hashes and weight offsets are recorded. Build commands are in build-commands.json. prepare-ffn.py obtains the inputs and tensor index; run-ffn.py performs the original full FFN controls; run-replay.py adds --down, --q8 and --bench. run-cpu-delta.py runs the independent projection comparison. analyze-ffn.py and verify-integrity.py validate and summarize preserved outputs. Each run requires a fresh output directory.

Large tensors, model weights, binaries, full outputs and Nsight traces remain in /home/sergei/_my_sync/llama_upstream_review/stage11/. raw-artifact-sha256.json identifies those data. The final profile was exported with Nsight Systems to ffn-ada-profile.sqlite before analysis. Initial attempts are excluded as recorded in excluded-attempts.json.

No production code or installed server changes. Prepared for review before commit.
