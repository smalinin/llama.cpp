# Stage 15 diagnostic artifacts

Stage 14 is committed as a4dc8f7fa. Stage 15 remains pending review. No production source, installed server, model or profile changes.

The non-perturbing capture preserves all95 Stage14 logits at widths1/2/4. Query input86 uses column2 of batch84 at width4. The first remaining difference is layer0 routed down/weight/sum on identical hidden, IDs and weights. Original Q3_K down and CPU rounded/FMA sum controls explain the local difference on Ada/Ampere. Scalar recomputation removes it locally but the full95 replay still differs at argmax19.

Reproduce sequentially from the local workspace artifact directory:

1. `python3 prepare-capture.py`, then `python3 run-chain-capture.py` and `python3 analyze-chain.py`. Output directories must not already exist.
2. `python3 prepare-weight-index.py`, then `python3 prepare-down-inputs.py`.
3. Compile `down-replay.cpp` with C++17/O2, the repository ggml include directory, and stage8/candidate-bin libs: `-lggml-cuda -lggml -lggml-base`. Use stage8/candidate-bin for both link search and rpath. Run `python3 run-isolated.py`.
4. `python3 prepare-components.py`, then `python3 run-components.py`. These component outputs have no model-capture reference; their raw max_vs_capture placeholder is excluded from curated results.
5. Compile `cpu-reduction.cpp` with `g++ -std=c++17 -O2 -ffp-contract=off -shared -fPIC cpu-reduction.cpp -o cpu-reduction.so`; run `python3 analyze-isolated.py`.
6. `python3 run-isolated.py --profile`. Export each nsys-rep with `nsys export --type sqlite`, then `python3 analyze-profile.py`.
7. `python3 prepare-candidate.py`, then `python3 run-down-candidate.py` and `python3 analyze-candidate.py`.
8. `python3 summarize-selections.py`, then `python3 verify-integrity.py`.

Model GPU jobs and isolated GPU controls must run sequentially. Runners verify the immutable Stage8 library snapshot and sanitize the same environment options as earlier stages. Sources and manifests preserve absolute paths in the user's workspace. `diagnostic-callback.h` is byte-identical to Stage14. The candidate caches live inputs before allocator reuse; it does not read dead hidden buffers at the output boundary.

The rejected expanded-capture sources and SHA controls are preserved for audit, with rejected prefixes. Their wide logits do not match Stage14 and are excluded from localization evidence.

Curated capture metadata and compact summaries are archived here. Large tensors, logits, binaries, raw SHA index and profiler traces remain in the original local workspace directory. `integrity.json` records the raw index SHA. `artifact-sha256.json` covers every archived file except itself. To archive, run `archive-review.py`; repo writes require the configured filesystem permission.
