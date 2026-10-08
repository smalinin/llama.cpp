# DeepSeek-V4.1 dense/sparse boundary controls

The Stage19 diagnostic header is reused byte-for-byte. This stage tests the CUDA attention threshold on fixed accepted histories with one sequence, F16 KV and the same six-GPU order. Production sources, installed server and profiles are unchanged.

The loaded model has indexer top_k2048 and SWA128. At these windows n_kv_max is2176. The MLA512 sparse predicate requires K>=max(4096,2*n_kv_max), so the actual transition is K4352, at absolute3328 for ratio1 and6657 for ratio2. K4096 remains dense. Source hashes and the predicate are recorded in `source-selection.json`; runtime masks and parameters are recorded independently.

1. `python3 prepare.py`, then `python3 run.py` and `python3 analyze-model.py`.
2. `python3 prepare-attention.py`, then `python3 run-attention.py` and `python3 analyze-attention.py`.
3. `python3 run-attention.py --profile`, then `python3 analyze-profile.py`.
4. `python3 verify-integrity.py`, then `python3 archive-review.py` after preparing the report and plan.

Run GPU jobs sequentially. Preserve existing output directories before rerunning. The replays reject existing output roots. Build commands, binary/source hashes, immutable library hashes and GPU UUIDs are stored in the manifests.

`sparse-replay` loads one model for thirteen contexts: prompt3033 with widths1/4; prompts3289/6617 with widths1/2/3/4; prompt6617 width4 without trace; and baseline95 widths1/4. Each new window predicts96 fixed token IDs. Prompts start with the existing23-token explain prompt and append repeated Stage19 token IDs to the requested length. These are teacher-forced diagnostic histories, including forced EOS tokens, not free completions or quality measurements.

Prefill batches are bounded by2048; the last two chunks are512/4. Trace adds no callback ask decisions. It captures FA0/2/20 near the window starts and boundaries. Captured FA outputs are after the existing diagnostic intervention. The source K/mask extent can therefore be wider than the per-query extent used to produce that output.

The frozen harness reuses `../stage18/attention-replay`. Twenty-two cases run twice on each of the model's CUDA0 and CUDA2 Ada boards and on Ampere. The owner device matters for matching model captures. Native/cropped inputs match their post-callback captures; the uncropped wide inputs are an isolated counterfactual. Padding adds masked zero rows. Setting n_kv_max=0 disables sparse at the same padded shape without changing the visible mask.

Nsight confirms the use_sparse template flag and mask compaction calls. All profiled outputs must match unprofiled hashes. Sparse and dense results are compared locally on each device; architecture-independent equality is not required.

Raw tensor bins, full logits, executables, runtime logs and Nsight files remain in this workspace. The review archive contains source, manifests, compact results, metadata and hashes. This callback duplicates work and performs CPU transfers, allocation and synchronization; it is not a production optimization. Expanded free DSpark generation, other cache states and production throughput remain open.
