# DeepSeek-V4.1 expanded diagnostic server correctness

The Stage19/20 callback is reused byte-for-byte. The expanded check finds an unsupported restored raw-KV layout; overall server compatibility remains open. Successful fresh completions are compared with native independently from failed cache requests. No production source or installed server is changed.

`prepare-build.py` compiles a copy of server-context.cpp and replaces only libllama-server-impl.so in an isolated snapshot. Compression ratios come from the loaded target model. The draft callback is disabled, interventions apply only to generation, FP precision is restored after successful decode, and cached routed-down inputs are cleared after successful batches. An exception ends the test process; exception cleanup is not certified for production use.

Run GPU jobs sequentially and preserve output directories:

1. `python3 prepare-build.py`
2. `python3 run-free-generation.py` (stops after the first N1 cache error).
3. `python3 prepare-build-extent.py`
4. `python3 run-followup.py` (completes fresh N1/N3 cases and records the expected cache errors).
5. `python3 analyze-free.py`, then `python3 verify-integrity.py`.
6. Prepare report/plan, then `python3 archive-review.py`.

The initial matrix is snapshot-off, integration-off, candidate-off, candidate-n1 and candidate-n3, with14 planned completions each: four previous prompts twice with n_probs5/0, and three long prompts fresh then cached. After the N1 dense-cache error, a new N1 process runs the two remaining fresh long prompts and a ratio2 cache probe. A new N3 process runs all11 fresh prompts and a ratio2 cache probe. The followup build only adds exception-time logging of FA shapes; arithmetic and callback ask decisions are unchanged. Each build has its own immutable library directory and hashes.

Greedy sampling, one sequence, context8192, F16 KV and the same six-GPU order are fixed. Short completions are bounded by256 tokens, long completions by128. Long prompts contain natural archive text followed by one HTML/SVG request, with thinking disabled. The rendered template is tokenized with special-token parsing. The first count-128 and last128 tokens give exactly3033/3289/6617 prompt tokens, removing only middle background. No EOS or continuation tokens are forced. These are the same final request at different context lengths, not three different quality tasks.

The wrapper observes the existing post-operation FA callback without adding ask decisions. Its r1/r2 counters classify each recomputed wide scalar query using the effective padded extent and sparse threshold established in Stage20. Predicate coverage uses only width>1 events: raw_limit is initialized by scalar_attention on wide batches, so width1 observer counts are excluded. These are predicate counters, not a new profiler measurement. Raw/compressed crop counts and decode ranges provide additional coverage evidence for fresh histories.

Control cache reuse is demonstrated by positive cache_n; fresh/cache equality is checked separately. Speculative cache requests fail the existing unsupported-cache-extent guard before generating a complete response. Do not disable the guard or count them as output comparisons. KV get_n_kv uses padded physical used_max_p1; state_read_meta allocates restored rows with find_slot. Absolute position alone does not establish a restored raw extent. Exception-time shapes and source hashes are saved for the next cache-policy fix.

Raw logs, full probabilities, executable/archive/object files and memory samples remain in the workspace. The review archive omits probabilities but preserves their hashes and the full raw-file index. General cache compaction, saved-session restore, multi-slot operation, other KV types, sampling distributions and production throughput are outside this bounded check. The callback duplicates GPU work and uses CPU transfers, allocation and synchronization; its timing is diagnostic.
