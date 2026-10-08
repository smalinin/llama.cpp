# Stage 17 free-generation checks

Stage16 is committed as 903560a8f. Stage17 is pending review. This directory contains an isolated diagnostic server, not an installed server update.

The shared diagnostic-callback.h concatenates the exact Stage16 candidate callback and the Stage15 cached scalar down helpers inside the same namespace. Server and target-only replay use this one header. prepare-build.py changes a temporary copy of server-context.cpp; only libllama-server-impl.so differs from the immutable Stage8 snapshot. Draft callbacks are disabled. Target precision is restored and cached down inputs are cleared after each decode. Prefill receives no interventions.

Reproduce sequentially in the original workspace artifact directory:

1. `python3 prepare-build.py`.
2. `python3 run-replay-control.py` and `python3 analyze-replay.py`. All95 logits at widths1/2/3/4 must match the Stage16 native SHA before continuing.
3. `python3 run-free-generation.py --configs snapshot-off integration-off candidate-off`.
4. `python3 analyze-free.py --controls-only`. Integration-off and candidate-off must retain all native token lists and content before continuing.
5. `python3 run-free-generation.py --configs candidate-n1 candidate-n3` and `python3 analyze-free.py`.
6. Compile `explain-replay.cpp` with the command in explain-build-manifest.json. The forced256 IDs are serialized from free-runs/snapshot-off/explain-greedy-2-response.json. Run `python3 run-explain-replay.py` and `python3 analyze-explain.py`.
7. `python3 verify-integrity.py` and `python3 archive-review.py`. Repository writes require the configured filesystem permission.

Output directories must not already exist. Run GPU jobs sequentially. The runners verify snapshot hashes, loaded-library maps and sanitized environment settings. Paths refer to the user's local model files and artifact directories.

Five configurations each run four prompts twice: n_probs5 and n_probs0, temperature0, seed1234, limit256, cache_prompt=false. These are probability-output controls, not identical-parameter timing repetitions. Compare complete token IDs, content and stop metadata up to EOS or the limit. Record speculative acceptance separately from output equality. Callback timings include original wide operations, recomputation, CPU copies and synchronization; they do not estimate production throughput.

The original Stage14 event prefix DS14_EVENT is retained for the existing log parser. At generation widths2..4 the expected per-call replacements are40*width for attention,40*width for upgate, and167*width total matmul results:80 HC,40 router,40 routed down and7 compressor projections. Width1 retains scalar arithmetic; prefill has zero interventions.

All baseline95 logits match native at widths1/2/3/4. All8 N=1 free answers match. N=3 matches6 of8; both explain answers diverge at index244. A separate native256 explain replay reproduces the same token choice at width3 without draft or rollback. Numerical differences start at output232 for width3 and233 for widths2/4, when the batch first includes absolute position256. A K/V padding boundary is a hypothesis for the next capture, not a proved cause. Broader float2d gives the same full logits as the narrow candidate on this history.

The curated archive keeps reproduction sources, source patch, manifests, response token IDs/content/settings/timings, callback events and summaries. Large binaries, full generated server-context.cpp, full probability arrays, logits, raw logs and raw-file-sha256.json remain local. Omitted probability arrays are marked in each curated response. integrity.json records the raw index hash; artifact-sha256.json covers all archived files except itself.
