# my_build upstream integration review

[Plan](PLAN.md) and [stage 0 report](STAGE0_REVIEW.md) approved by the user before implementation.

This directory records baseline requests, responses, metrics, model metadata and reproduction scripts. Source baseline: a3a7dacd92b4a01928cbb71ecc7ddfe4e5cef683.

Large build/test logs, per-second memory samples and the preserved server binaries remain in `/home/sergei/_my_sync/llama_upstream_review/stage0/`. Paths in recorded commands refer to that local artifact directory. Binaries and model files are not checked into Git.

[Stage 1 report](STAGE1_REVIEW.md) and its recorded requests/results were approved by the user. Stage 1 source changes and this report are included in the same commit. Raw upstream patches, source.patch, logs, memory samples and preserved binaries remain in `/home/sergei/_my_sync/llama_upstream_review/stage1/`.

[Stage 2 report](STAGE2_REVIEW.md) was approved by the user and committed as c788620d1. The row-order and recurrent-gather changes are validated by the recorded harnesses and real-model requests. Strict scheduler checks and the GLM5NEXT unified-KV multi-slot pool-key assertion remain limitations; see the report. Invalid server controls caused by absolute RUNPATH were excluded and repeated with isolated libraries. Logs, binaries and raw reference dumps remain in `/home/sergei/_my_sync/llama_upstream_review/stage2/`.

[Stage 3 report](STAGE3_REVIEW.md) was approved by the user and committed as 0a1ad5457. Failed state restore cleanup and exact rotation metadata are adapted to the local caches. Session/sequence format versions are now 11/4, so old state files must be recreated. Existing target-only slot files do not preserve draft/speculative state; native continuation and speculative fallback are documented separately. DeepSeek native continuation differs after restore on both old and new libraries; this criterion remains unresolved. Logs, binaries and raw state files remain in the local stage3 artifact directory.

[Stage 4 report](STAGE4_REVIEW.md) was approved by the user and committed as cc7bfba3b. The CUDA RMS_NORM/SCALE fusion and thin FP16/BF16 matmul changes were checked again after the GPU replacement.

[Stage 5 report](STAGE5_REVIEW.md) was approved by the user and committed as 11638b685. Four-head Lightning Indexer support is available; the experimental Qwen fused QSA remains opt-in.

[Stage 6 report](STAGE6_REVIEW.md) is prepared for review before commit. All ten repeated answers in each of three GLM-5.3 modes match, including server restarts. The original sampling mismatch was not reproduced on the current hardware; its cause remains unknown. The user then requested proceeding to the final DSpark stage; Stage 6 remains uncommitted.

[Stage 7 report](STAGE7_REVIEW.md) is prepared for review before commit. The DSpark length/confidence sweep did not establish a speedup. A validated target-only replay reproduces the 34th-token greedy divergence by changing decode width without draft or rollback; the responsible operation is not yet localized. The installed server and production DSpark sources are unchanged. Raw traces, float logits and logs remain in the local stage7 artifact directory.

[Stage 8 report](STAGE8_REVIEW.md) is prepared for review before commit. Selectively requantizing the compatible MXFP4 DSpark experts to Q4_K did not improve throughput and added 405 MiB. CUDA MUL_MAT now respects explicitly requested FP32; regression tests and real-router replay validate that correction. An additional HC precision experiment still diverged during free greedy generation and was removed from the working model graph. Full target-logit compatibility and a stable DSpark speedup remain open. The installed server and profiles are unchanged.

[Stage 9 report](STAGE9_REVIEW.md) is prepared for review before commit. A new server-matched native prefix and non-perturbing tensor capture localize another source of batch dependence to BF16 compressed-KV projections. Decode-only FP32 controls preserve the tested argmax prefix but leave large logit differences; no additional model-graph flags are retained. DSpark N=1 still diverges on all four prompts and does not establish a general speedup. All 42 HTTP requests and 49 replays completed successfully. Full target-logit compatibility remains open; the installed server and profiles are unchanged.

The user approved committing the reasoning-budget fix, CUDA FP32 fix and Stage 6-9 reports as 05ea5d72f, then continuing the remaining DeepSeek-V4.1 diagnosis. The unresolved compatibility and performance criteria remain open. Earlier report statements about review status describe their preparation time.

[Stage 10 report](STAGE10_REVIEW.md) was approved by the user and committed as 1ffcefe92. All 21 model replays and 17 logit SHA controls passed. Fixed-input attention varies numerically with batch and architecture; two real Q6_K Q-projections instead match exactly across widths 1/2/4 on identical inputs. CPU Q8_1 reference confirms amplification of small input differences in those projections. Scalar attention plus FP32 floating projections still does not restore general greedy compatibility. No production patch is introduced; the installed server and profiles are unchanged.

The user approved committing Stage 10 and continuing the remaining layer-2 FFN diagnosis. General greedy compatibility and DSpark speedup criteria remain open.

[Stage 11 report](STAGE11_REVIEW.md) was approved by the user and committed as dc7cdcef7. The isolated layer-2 FFN reproduces all six captured width/architecture outputs exactly. On identical inputs, routed up/gate arithmetic crosses two Q8_1 rounding boundaries in expert 217; frozen-down controls and CPU projection explain the local amplification with a residual below 3e-8. Nsight identifies the scalar/batched reduction paths. Full scalar FFN is locally estimated to cost 1.46-2.04 times batched execution; this is not a whole-model performance result. No production patch is introduced. General greedy compatibility and DSpark speedup remain open; the installed server and profiles are unchanged.

The user approved committing Stage 11 and continuing the narrow routed up/gate arithmetic experiment. General greedy compatibility and DSpark speedup remain open.

[Stage 12 report](STAGE12_REVIEW.md) was approved by the user and committed as fec769528. Scalarizing only routed up/gate/SwiGLU makes fixed-input hidden bit-identical and reduces local FFN differences to about 3e-8, with 11.5-34.4% isolated FFN overhead at widths 2/4. All 12 whole-model replays and 8 SHA controls pass, but general greedy compatibility still fails: combined FP32/scalar-attention/upgate retains an argmax difference at index19 for width4 and increases logit max/RMS versus the previous FP32/attention control. The candidate remains diagnostic; no production patch or installed server change.

The user approved committing Stage 12 and continuing the remaining HC/router/FFN diagnosis after the attention/upgate controls. General greedy compatibility and DSpark speedup remain open.

[Stage 13 report](STAGE13_REVIEW.md) is prepared for review before commit. Non-perturbing captures localize the first remaining differences to layer-0 router at width2 and HC mixes at width4, on identical inputs. The isolated HC post controls for layers0/2 match across widths on Ada/Ampere. All 12 model replays, six reference SHA controls, 88 isolated cases and 44 profiler cases pass their documented checks. Scalar HC+router added to the Stage12 diagnostic control preserves all65 native argmax values at both widths and makes the full width2/width4 logits bit-identical to each other. Large scalar logit differences remain. Free DSpark generation and actual throughput are not yet tested for this candidate; no production patch or installed server change.

The user approved committing Stage 13 and continuing the diagnostic HC/router combination in free greedy DSpark generation. General compatibility and speedup remain open.
