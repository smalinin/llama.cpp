# Stage 10 review artifacts

21 model replays, 14 fixed-input attention variants and 72 projection variants completed successfully. Isolated GPU variants were computed twice. See ../STAGE10_REVIEW.md and integrity.json for scope and limitations.

Run manifests record exact commands, input/source/binary hashes, model paths and GPU order. build-commands.json records rebuild commands. Scripts depend on the preserved local Stage 5/7/8/9 artifacts and Stage 8 candidate libraries; large tensor data, extracted weights, logits, binaries and full logs remain in the workspace. raw-artifact-sha256.json identifies those local data files.

initial-attempt.json records the interrupted first harness attempt, excluded from conclusions. A CPU reference attempt that collided with its binary path was also excluded and repeated in a separate output directory.

No production graph changes are included in Stage 10. Exact greedy compatibility and a DSpark speedup remain open. Prepared for review before commit.
