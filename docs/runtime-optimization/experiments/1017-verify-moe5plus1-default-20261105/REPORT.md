# MoE5+1 default promotion, 2026-11-05

The user requested default ON after the [controlled performance gate](../1017-verify-moe5plus1-performance-20261105/REPORT.md). `LLAMA_DSV41_VERIFY_MOE_WIDE` now uses the existing default-on environment helper: unset/1 enables, 0/off disables. The graph eligibility,5+1 grouping, tensor views, CUDA selectors, masks, state updates and floating batching limits are unchanged.

The [private build](artifacts/build.json) and [four joint no_alloc checks](artifacts/checks.json) verify default versus explicit1, and explicit0 versus off. Reserved allocations and routes match the previously measured explicit ON/OFF candidate; all15360 expert bundles stay on GPU. This is a default-selection check, not a new full correctness/performance run. Prior observed logits/state/rollback and OFF/ON/OFF speed results remain the evidence for this same mechanism.

Scope remains DeepSeek41 six-query/K5 text/P1/Q8 CSA+LID/full CUDA experts/pipeline with contiguous separate Q3_K/Q4_K tensors and no eval callback/warmup. Other cases keep the existing fallback. This changes no global five-query floating guard or other model's graph. Historical uninstrumented prefill/rollback failures remain unresolved and preserved; default promotion by request does not declare their root cause fixed.

The private build and default checks are complete. A separate normal llama-server build is in progress; it is rebuilding CUDA objects as well. Its eventual outcome is separate from the completed selection checks in this commit. Next is Phase B floating sample-grid6 behind its own initially-OFF flag; this promotion does not enable that candidate.
