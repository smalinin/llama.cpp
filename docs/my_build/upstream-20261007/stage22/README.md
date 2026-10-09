# DeepSeek-V4.1 restored raw-KV extent diagnostic fix

Stage21 found HTTP500 after prefix-cache restore: raw K has256 rows at absolute6617, while physical storage is768. The previous absolute-position rule requests768. This stage keeps the existing numerical interventions and supplies a physical per-query raw extent instead.

`raw-layout.h` reads the single-sequence raw SWA cache before decode, obtains the full batch slots through the const find_slot API, and simulates each prefix on copied cell positions. It mirrors the overwrite/purge rule from apply_ubatch, then rounds physical used_max_p1 to256 and clamps to storage size. It changes no live cells or heads and touches no GPU data. At FA0 the plan's final extent and every physical input index are checked against the real cache. The existing -inf mask-tail guard remains active for every crop.

The general callback accepts an optional raw_query_extents vector; the old fresh-history formula remains the fallback for old harnesses. The Stage22 model/server harnesses always populate the vector during generation. Compression ratios and per-query compressed extents retain the tested policy. This does not provide arbitrary multi-sequence compaction or saved-session guarantees.

The target-only replay saves a partial checkpoint four tokens before the prompt end, completes a128-token accepted history, restores the checkpoint, reevaluates the last four prompt tokens and measures a128-token fixed prefix. It compares full native-width1 logits with widths2/3/4 on prompts3033/3289/6617. The fixed IDs come from Stage21 native free generation. Draft and sampling are absent in this replay. Baseline95, the Stage20 ratio2 sparse window and the Stage19 history1024 retain previous SHA controls.

Three restored1024 fixed-history cases at widths1/3/4 checks physical extents256/512/768 and a raw ring wrap. These IDs come from the Stage19 explain history on the Stage21 ratio2 prompt; they test arithmetic, not free-answer quality.

Reproduction: build the model harness with the command recorded in replay-build-manifest.json, run run-replay.py --cases cases-all.txt, then analyze-model.py and analyze-long.py. Run prepare-build.py before run-free-generation.py and analyze-free.py. Use Python3; the analyzers require only the standard library. Both source patches and exact build/link commands are retained.

All GPU jobs must run sequentially. Model and server artifacts use the immutable Stage8 libraries. Server builds replace only libllama-server-impl.so in this stage's separate directory. The production sources, installed server and profiles are unchanged. This callback still duplicates work and is not a production speed optimization.

Two harness setup mistakes are retained separately and excluded: an unassigned ubatch seq_id pointer, and omission of the server tail trim after partial restore. The final harness restores PARTIAL_ONLY then calls llama_memory_seq_rm at the checkpoint position before re-prefill, matching server-context.cpp.

Final recorded outcome: all56 HTTP responses and all six cached speculative requests match native. All12 restored128 replays and four prior full-logit SHA controls are exact. The restored1024 stress test is a documented negative result: wide3/4 match each other, but differ from scalar at row751 despite exact physical traces and argmax. verify-integrity.py checks these recorded positive and negative outcomes; it does not certify general target-logit compatibility. Review is pending before commit. The next stage must localize query7367 before production CUDA/performance claims.
