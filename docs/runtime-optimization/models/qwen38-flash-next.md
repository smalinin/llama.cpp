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

MTP Top-K reuse is restricted to a context configured for one sequence; multi-sequence contexts recompute selection. Shared target embedding/head and dense MTP fallback remain. Actual mmproj image requests, real MTP acceptance/speed, full PPL/task evaluation, larger real contexts and other quantizations are unmeasured. See the experiment and local raw archive for source/build hashes and limits.
