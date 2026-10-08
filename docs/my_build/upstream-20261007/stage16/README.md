# Stage 16 diagnostic artifacts

Stage15 is committed as d0060fd95. Stage16 is pending review. No production source, installed server or profile changes.

Adding scalar compressor KV/gate to the Stage15 diagnostic combination makes all95 full logits bit-identical to native at widths1/2/4. A broader scalar floating2D control gives the same result. This is one teacher-forced history; free DSpark generation, width3 and performance remain unverified.

The layer2 capture preserves all Stage15 logits. Q and masks match. Small compressor projection differences change three F16 components in one visible K/V row at input2. Replacing only that row reproduces the wide attention output on both tested architectures. All12 isolated BF16 projection capture controls are exact. Direct scalar down graphs cost1.56-1.82 times the corresponding wide graph locally, excluding callback duplication and copies.

Reproduce from the original workspace artifact directory with the immutable Stage8 snapshot:

1. Run `python3 prepare-capture.py`, `python3 run-chain-capture.py`, and `python3 analyze-chain.py > chain-overview.json`.
2. Compile `capture-attention.cpp` with the exact command in `attention-build-manifest.json`. Run `python3 run-attention-capture.py` and `python3 analyze-attention.py`.
3. Run `python3 prepare-down-cost.py`, `python3 run-down-cost.py`, and `python3 analyze-cost.py`.
4. Run `python3 prepare-compressor-candidate.py`, `python3 run-compressor-candidate.py`, `python3 analyze-candidate.py`, and `python3 analyze-candidate-boundaries.py`.
5. Run `python3 prepare-isolated.py`, `python3 run-compressor-isolated.py`, and `python3 analyze-isolated.py`.
6. Run `python3 verify-integrity.py` and `python3 archive-review.py`. Repository writes need the configured filesystem permission.

Output directories must not already exist. Full-model and isolated GPU jobs must run sequentially. Runners sanitize environment options and verify32 immutable library hashes. Loaded-library maps are checked. Sources and commands preserve local absolute paths; model weights and libraries are not redistributed in Git. Python uses stdlib plus the existing Stage5 metrics.so.

The base diagnostic-callback.h is byte-identical to Stage15. candidate-callback.h adds only compressor selection. Cached down inputs remain captured while live. FA JSON scale has six decimal digits; the isolated replay uses the exact original C++ scale formula, rather than that rounded value.

Curated sources, summaries, manifests, tensor metadata, row argmax and small frozen projection inputs are archived here. Large tensor/logit files, binaries, model weights, raw logs and raw-file-sha256.json remain local. integrity.json records the raw index SHA. artifact-sha256.json covers all archived files except itself. The archive script copies the current report and PLAN.md as well.
