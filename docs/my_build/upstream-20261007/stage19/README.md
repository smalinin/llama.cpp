# DeepSeek-V4.1 later KV padding and SWA controls

The Stage18 callback handles only the first padding boundary. `legacy-callback.h` is that header with a renamed namespace. `general-callback.h` derives per-query raw and compressed extents from the absolute position, actual raw storage and model compression ratios. It checks every removed mask entry. The scalar reference is unchanged. This is a diagnostic combination with the earlier precision, attention, FFN, HC/router and compressor interventions.

The fixed 1024-token history concatenates native response token IDs. The longer-prompt history contains the original 95-token baseline followed by 417 IDs from that concatenation. These are teacher-forced inputs, not free continuations or quality measurements. Input index i decodes absolute position prompt_length+i and predicts output index i+1.

Use immutable `../stage8/candidate-bin` libraries and the recorded GPU UUID order. Run all GPU jobs sequentially. Preserve completed output directories before rerunning; the replay executables reject existing output roots.

1. `python3 prepare.py`, then `python3 build-replay.py`.
2. `python3 run.py`, then `python3 analyze.py` and `python3 analyze-ring.py`.
3. `python3 prepare-attention.py`, `python3 run-attention.py`, then `python3 analyze-attention.py`.
4. `python3 build-schedules.py`, then `python3 run-schedules.py` and `python3 analyze.py`.
5. `python3 verify-integrity.py`, then `python3 archive-review.py` after preparing the review report and plan.

`boundary-replay` loads one model for ten replay contexts: old fixed widths 1/3/4, old width4 without trace, new fixed widths 1/2/3/4, and baseline95 widths1/4. Trace uses the callback's existing ask decisions; it adds no graph boundaries. Focus captures record FA0/2/20 operands near absolute256/512/768/1024.

`schedule-replay` loads one model for five contexts: scalar with three snapshot planes, mixed append widths1/4/2/3, mixed widths4/3/2/4 with partial rejection near boundaries, and widths1/4 on a 1655-token prompt plus a 512-token fixed continuation. Partial removals reject at most three tokens and must succeed. Accepted logits are recorded once; all attempts are retained separately. No draft model or sampling is used in these replays.

`run-attention.py` reuses the Stage18 frozen-attention executable, recording its binary and source hashes. Ten cases run on each of Ada and Ampere, twice each: captured native/wide, native padded with hidden zeros, wide cropped, and native with wide n_kv_max. Only local same-device results are required to match; Ampere need not equal the Ada model capture.

The tested attention lengths stay below the CUDA sparse threshold. Sparse dispatch, arbitrary cache compaction/restore, multiple sequences and production throughput remain unverified. Do not use this callback as a production optimization. It duplicates graph operations and performs CPU transfers, allocation and synchronization.

Raw logits, tensor bins, executables and full runtime logs stay in this workspace. The Git archive contains sources, manifests, compact results, trace and capture metadata, completion counts and hashes. Production sources, installed server and profiles are unchanged.
