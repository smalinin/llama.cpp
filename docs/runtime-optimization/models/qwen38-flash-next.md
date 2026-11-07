# Model passport: Qwen3.8-Flash-Next Q8_0

Date: 2026-11-06. Working branch: `my_glm53_flash_1`, base `a4968c984`. [Port experiment](../experiments/qwen38-port-20261106/REPORT.md).

| Field | Evidence |
| --- | --- |
| Weights | `/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf`; six unchanged shards |
| Architecture | `qwen4exp`, 48 trunk layers, 36 GDN / 12 QSA, hidden 2560, 10 of 512 routed experts |
| QSA | Four 128-wide index heads, budget 4096, pool size 4, raw/pooled F16 index rows, cell-order grouping and live tail |
| PLE | CPU lazy mmap table, n-gram embeddings; unchanged files |
| Native workload | 21000 saved text IDs, 64 teacher-forced decode steps, clear and recompute each repetition |
| Settings | ctx262144, B2048/U512, P1, unified F16 KV, FA ON, threads20, mmap/lazy AUTO, fit target512 MiB |
| Hardware | Five 48 GiB RTX4090/4090D and one 24 GiB RTX3090; Threadripper PRO3945WX; CUDA13, SM86/89 |
| Placement | All layers/expert weights on GPU, 9/9/9/9/9/3; pipeline enabled, four copies |
| Result | PP1911.846 ->4499.916 (+135.3702%), TG53.030 ->63.751 (+20.2169%); three warm native repetitions |
| Fresh upstream | PP4819.886 / TG73.122; local port remains 6.6385% /12.8156% below it |
| Consumers | Plain-text native baseline; synthetic MTP/shared head, rollback, snapshots and M-RoPE tests |

GDN MMA is restricted to final-state-only K=1 and supported geometry/devices. Multiple snapshots use the old AR path. Our GLM cache layout and established MMQ selectors remain. Allocator dependencies are retained across zero/single/many-row reductions, even when execution uses its fallback; this preserves the worst-case PP plan across decode.

MTP Top-K reuse is restricted to a context configured for one sequence; multi-sequence contexts recompute selection. Shared target embedding/head and dense MTP fallback remain. Actual mmproj image requests, full PPL/task evaluation, larger real contexts and other quantizations are unmeasured. See the experiment and local raw archive for source/build hashes and limits.

## Real shared-MTP prefill follow-up

[Experiment](../experiments/qwen38-mtp-20261106/REPORT.md), base c83afd9d3 plus a working-tree patch: ordered asynchronous hidden-state export and stable rollback graph nodes remove repeated reserves. With per-logical-batch synchronization and real shared-draft catch-up, warm PP1712.881 ->2803.962; matched plain control3498.111. Two cycles per arm, not medians. The server follow-up below measures real acceptance/TG; images remain unmeasured. State/export tests pass on CPU/Ada/Ampere; Qwen/GLM architecture checks pass. The upstream b12138 failures for shared loading and a zero-compression MTP indexer are explained in the report; our compatibility fixes were already present.

## Real server generation follow-up

Same final binary, text21K plus256 generated tokens, sampling1.0/0.95/20/0.05, seed1234, cache disabled, one cold plus three warm requests per arm. Plain controls before/after give PP4275/4250 and TG59.58/58.88. Shared MTP maximum2 gives PP2849/TG56.67, acceptance66.96%; maximum1 PP2904/TG59.07, acceptance75.81%. No MTP speed gain is established for this corpus. Maximum1 TG is near plain within variation, but overall requests still cost more because of prefill. Real image integration and formal quality gates remain open; no defaults/configuration changed. See the same [experiment](../experiments/qwen38-mtp-20261106/REPORT.md) for ranges and controller cleanup history.

## Original MTP compatibility and fixed drafting

[Original comparison](../experiments/qwen38-original-mtp-20261106/REPORT.md): stock b12138/e14 abort with the supplied ordinary MTP metadata. An independent copy changing only compression ratio element48 from0 to4 runs with the unchanged original binary: PP3244/TG72.52 versus original plain4522/70.03. Our exact-same-draft default gives2867/62.58; LLAMA_MTP_ADAPTIVE=0 gives2876/69.26, with wide TG variation54.84-70.67. Our build with the unchanged shared draft and adaptive OFF gives2879/62.87 versus prior default2849/56.67. No runtime/default/model-source change; full/shared tensor equivalence and formal quality remain unmeasured.


## Plain TG optimization after Strata profiling, 2026-11-07

[Experiment](../experiments/qwen38-strata-port-20261107/REPORT.md), base6d034e6b9, measured local defaults. Text21159 IDs from the saved Strata renderer,512 sampled outputs, top-K attention2048, ctx262144/B2048/U512/P1/F16 KV/threads20, no MTP. Same6-GPU9/9/9/9/9/3 placement: clean TG62.681->71.375 (+13.87%), all16 responses exact. First-request PP3679->4094, repeated PP4273/4212/4512->5598/6492/7035; PP history matters. Accepted: layer-local HC norm, existing-key reuse by immutable CUDA graph UID, bounded MoE/shared stream overlap, fused Q8 down/reduction for640x2560/512 experts/10 selected, and four-warp Q8 gate/up grouping for2560x640/10 selected. Raw full logits/states/restores/failure-retry and synthetic embeddings exact; default-on repeat and existing sparse/save-load tests pass. Graph-key fallback is shared; model/geometry guards scope the other changes. GDN value heads48, HC down/up/inject Q8_0, norm F32.

Peak total NVML141.645GiB, min sampled free13626MiB, six cards. Four fast GPU screen139.125GiB, TG72.43-72.86, but output differs at token7 and first-request PP3417 is slower, so no preset/device-list change. Strata controls79.1 use INT8 KV/prepared HC BF16 plus Nsight injection; exact cross-runtime equivalence and matched native parity are not established. Async input staging, typed metadata packing and fused GDN norm/gate are removed with patches; no established CPU thread-count gain. New flags default ON with individual0 fallback; extended down support remains opt-in outside the narrow Qwen case. Real media, formal PPL, mixed offload/multisequence and MTP remain outside this experiment. No commit yet.
