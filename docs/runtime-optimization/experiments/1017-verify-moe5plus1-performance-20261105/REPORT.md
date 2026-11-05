# MoE5+1: controlled target cost and native throughput, 2026-11-05

Status: the observed cost/native controls PASS. Native OFF/ON/OFF keeps all 27 output arrays and draft proposed/accepted counters exact. Six-query target latency falls 20.26% at 8K and 15.84% at 88K. Server TG gains are smaller and prefix-dependent, as measured below. This is bounded performance evidence for the existing opt-in, not a default promotion or a resolution of the earlier uninstrumented prefill/rollback failures.

## Scope and identity

Runtime commit e69c07128; documentation HEAD afc37abd8 during this experiment. Reused private libllama SHA256 `9c440a1860f1f456018b514b5974f2dd99385bd1d7e3756234f447799d392ff3`, model.cpp SHA256 `17c3c983d276b3a7b5fd1bb3b751ed627d4f7bd98db04fd424a119b842f78b54`. Both arms load this same library; only startup `LLAMA_DSV41_VERIFY_MOE_WIDE=0/1` changes. Production libraries, runtime source, defaults and presets are unchanged. `LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=0` in every arm isolates this mechanism.

New host: Threadripper PRO 3945WX, 192 GB RAM, six UUID-bound Ada/Ampere cards in recorded PCI order. Frozen target Q3_K-AntirezLike with Q4_K down experts in layers 0..4, MXFP8 CPU Engram, Q8_0 K/V, TopK2048, ctx92160, B8192/U512/P1/threads12. All 15360 expert bundles remain on GPU; expert cache/staging/backfill OFF, fitOFF. DSpark K5/full/p-min0.5, same mixed-weight sidecar whose filename says BF16, draft on CUDA5/CUDA0. Text-only, no mmproj; no callback/CUPTI/interposer in timed runs. Input IDs, expanded arguments/environment, device map, model identities and mapped library evidence are retained in the manifests. Historical `config.source_head` is inherited from the input recipe; use `width-results.head`, source hashes and mapped private module for this run's identity.

See [protocol](PROTOCOL.md), [build identity](artifacts/runtime-build.json), [frozen config](artifacts/config.json), [admission](artifacts/admission.json) and [input IDs](artifacts/inputs.json). No Maya speed comparison or old-platform timing is reused here.

## Raw target cost

The byte-identical [existing width harness](artifacts/width-build.json) runs fresh OFF then ON with the same restored prefix state and six fixed proposal IDs, widths 1..6 in ascending and reverse passes. Each pass has five measured repetitions per width, plus retained warmups. Timer includes target decode and synchronization, but excludes snapshots, draft processing, feature bridge, raw output transport and hashing. The draft is registered; this raw-logit path differs from the server GPU sampler.

| Prefix tokens | Target queries | OFF median ms | ON median ms | Latency change |
| --- | --- | --- | --- | --- |
| 8192 | 1 | 24.266 | 24.063 | -0.84% |
| 8192 | 2 | 30.129 | 30.147 | +0.06% |
| 8192 | 3 | 34.816 | 34.846 | +0.09% |
| 8192 | 4 | 39.067 | 39.048 | -0.05% |
| 8192 | 5 | 43.073 | 43.053 | -0.05% |
| 8192 | 6 | 68.571 | 54.680 | -20.26% |
| 90112 | 1 | 24.300 | 24.334 | +0.14% |
| 90112 | 2 | 31.703 | 31.710 | +0.02% |
| 90112 | 3 | 36.442 | 36.436 | -0.02% |
| 90112 | 4 | 42.222 | 42.250 | +0.07% |
| 90112 | 5 | 45.997 | 46.016 | +0.04% |
| 90112 | 6 | 68.162 | 57.361 | -15.84% |

All 156 observations per arm have matching full-logit hashes/first-row checks; 24 sampled state checkpoints per arm and prefix/proposal identities match. All 120 measured observations per arm reuse the frontend graph. Widths 1..5 are negative controls and show no material gain. Keep all warm/cold/variable rows; no outlier removal. No new CUDA-instantiation counters were inserted into this timing. [Raw manifest](artifacts/width-results.json), [OFF observations](artifacts/width-off/results.json), [ON observations](artifacts/width-on/results.json), [summary](artifacts/width-summary.json).

The prior [same-prefix counter diagnostic](../1017-verify-shared-prefix-20261105/REPORT.md), using the identical private runtime, shows MoE nodes 720->240 and backend submissions 113->57 at 88K. This clean width run separately records splits 114->58 at 8K and 113->57 at 88K. Floating matmuls remain 950 in the prior diagnostic. Node/call reductions are not kernel speedup ratios. Direct6 remains rejected by its Ampere Q3_K MMVQ-to-MMQ dispatch change; 5+1 preserves the tested scalar arithmetic.

## Native OFF/ON/OFF

Three fresh servers, each with one warmup and two measured requests per prefix. Clear prompt reuse before every request; require cache_n0 and full prompt_n. Each response generates 256 greedy tokens with identical GPU sampler/K5/full/p-min0.5. Warmups are excluded from speed but included in exactness checks. OFF baseline is the arithmetic mean of four measurements from both neighboring OFF arms; ON is the mean of two. This is a short controlled screen, not a confidence interval or a long workload benchmark.

| Prefix tokens | OFF TG t/s | ON TG t/s | TG change | OFF PP t/s | ON PP t/s | PP change |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 41.55 | 42.86 | +3.15% | 724.43 | 732.07 | +1.05% |
| 8192 | 45.64 | 48.60 | +6.49% | 1416.40 | 1420.51 | +0.29% |
| 90112 | 60.90 | 71.51 | +17.42% | 1447.88 | 1455.98 | +0.56% |

TG uses 255 timed intervals for 256 generated tokens, as in the existing runner: 255000/predicted_ms. The response's HTTP predicted_per_second uses 256000/predicted_ms and is about 0.39% higher; do not mix these conventions. PP is prompt_n*1000/prompt_ms. No prefill algorithm changes here; small PP differences are retained as controls, not attributed to MoE6 grouping.

| Prefix tokens | OFF before TG | OFF after TG | OFF drift | OFF range | ON range |
| --- | --- | --- | --- | --- | --- |
| 512 | 41.48 | 41.63 | +0.37% | 41.47..41.65 | 42.81..42.91 |
| 8192 | 45.61 | 45.67 | +0.13% | 45.45..45.77 | 48.48..48.72 |
| 90112 | 60.87 | 60.92 | +0.08% | 60.80..61.05 | 71.39..71.63 |

Every warmup and measured output array/hash and draft count matches across all arms/repeats, not merely the final greedy token. Exact counters are:

| Prefix tokens | Accepted/proposed draft tokens | Acceptance |
| --- | --- | --- |
| 512 | 131/214 | 61.21% |
| 8192 | 161/242 | 66.53% |
| 90112 | 198/234 | 84.62% |

These are aggregate acceptance rates, not per-position acceptance. K5 is a maximum: p-min0.5 can emit fewer proposals, so not every verification call has six queries. Consequently the raw six-query latency reduction does not translate into the same server TG gain. The three prefixes have different acceptance and content; compare OFF versus ON within a prefix, not their absolute TG as a context-scaling curve. [Raw server manifest/responses](artifacts/native-results.json), [independent summary/checker](artifacts/native-summary.py), [summary](artifacts/native-summary.json).

## Memory, lifecycle and correctness limits

Fresh joint OFF/ON no_alloc confirms identical placement and predicts minimum 1265.31 MiB after the recorded same-context measured overhead plus 64 MiB guard. Actual sampled width minima are OFF 1397.125 MiB and ON 1401.125 MiB. Actual native minima are off-pre 1329.250 MiB, on 1331.250 MiB, off-post 1329.250 MiB; global 1329.250 MiB. Every card stays above the 512 MiB floor. All five timed GPU processes exit 0, without forced kill, OOM, CUDA allocation errors or compute-buffer expectation warnings. Only experiment-owned servers are stopped. [Final identity/lifecycle checks](artifacts/final-checks.json) confirm frozen inputs, source and production modules and no remaining GPU compute processes.

[Earlier correctness evidence](../1017-full-precision-forensic-20261105/REPORT.md) passes 12 text/C++ groups, logits/taps/state, full/partial/sequence restore, rollback/append and two injected seed failures/retries; shared-prefix isolation separately passes 117 target/18 draft cases. Original NLL128 (4-query chunks), delta 0, does not activate the six-query branch; its full-logit/state checks do. Nine raw OFF first 8K controls also pass. The earlier uninstrumented rollback and first 8K mismatches remain unresolved and preserved in the [prototype report](../1017-verify-moe5plus1-20261105/REPORT.md). These controlled successes do not establish universal determinism or a root-cause fix.

## Decision and next step

Keep `LLAMA_DSV41_VERIFY_MOE_WIDE` opt-in/defaultOFF. Performance is now measured with matching controls for this fixed Q3/Q8/K5 text/P1/full-GPU setup. Other quants/GPUs, F16 KV, CPU expert cache/staging/spill, media, parallel, auto-fit262K, K3/K4 and interaction with prefill deferral need separate evidence; their support is not certified by this screen. No production module was replaced and no automatic merge/default promotion follows from this report.

[Phase B](../../designs/1017-six-query-moe-float.md) is next: isolate six-sample floating GEMV batching without changing reduction/weight-sharing kernels, check it alone and with 5+1. Phase C shared-weight MMVF6 remains separate. The existing floating 950 matmuls provide a concrete residual target; do not count the present MoE gain a second time.


## Subsequent default promotion by request

After this completed measurement, the user requested default ON. [Promotion evidence](../1017-verify-moe5plus1-default-20261105/REPORT.md) records that separate change and its checks. Frozen manifests here still describe the original explicit0/1/0 runs and defaultOFF source identity; they have not been rewritten as tests of the promoted module. Historical failures remain unresolved.
