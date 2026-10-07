# my_build upstream integration review

[Plan](PLAN.md) and [stage 0 report](STAGE0_REVIEW.md) approved by the user before implementation.

This directory records baseline requests, responses, metrics, model metadata and reproduction scripts. Source baseline: a3a7dacd92b4a01928cbb71ecc7ddfe4e5cef683.

Large build/test logs, per-second memory samples and the preserved server binaries remain in `/home/sergei/_my_sync/llama_upstream_review/stage0/`. Paths in recorded commands refer to that local artifact directory. Binaries and model files are not checked into Git.

[Stage 1 report](STAGE1_REVIEW.md) and its recorded requests/results were approved by the user. Stage 1 source changes and this report are included in the same commit. Raw upstream patches, source.patch, logs, memory samples and preserved binaries remain in `/home/sergei/_my_sync/llama_upstream_review/stage1/`.

[Stage 2 report](STAGE2_REVIEW.md) was approved by the user and committed as c788620d1. The row-order and recurrent-gather changes are validated by the recorded harnesses and real-model requests. Strict scheduler checks and the GLM5NEXT unified-KV multi-slot pool-key assertion remain limitations; see the report. Invalid server controls caused by absolute RUNPATH were excluded and repeated with isolated libraries. Logs, binaries and raw reference dumps remain in `/home/sergei/_my_sync/llama_upstream_review/stage2/`.

[Stage 3 report](STAGE3_REVIEW.md) is prepared for review before commit. Failed state restore cleanup and exact rotation metadata are adapted to the local caches. Session/sequence format versions are now 11/4, so old state files must be recreated. Existing target-only slot files do not preserve draft/speculative state; native continuation and speculative fallback are documented separately. DeepSeek native continuation differs after restore on both old and new libraries; this criterion remains unresolved. Logs, binaries and raw state files remain in the local stage3 artifact directory.
