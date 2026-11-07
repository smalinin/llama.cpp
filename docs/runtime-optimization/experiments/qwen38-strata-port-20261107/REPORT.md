# Qwen3.8-Flash-Next: plain decode optimization after Strata profiling

Date: 2026-11-07. Branch: `my_glm53_flash_1`, base `6d034e6b9d1a023557190c85252689dcde841337`.
Status: measured working-tree defaults, rebuilt locally; no commit, preset change or MTP changes in this experiment.

## Result and scope

On the fixed six-GPU text workload, clean native TG improves from median62.681 to71.375 tokens/s (+13.87%). All16 warmup/measured512-token response arrays match the original baseline exactly. First-request PP improves3679 to4094 tokens/s. Repeated PP is history-dependent: baseline4273/4212/4512, candidate5598/6492/7035; these are request ordinals after one warmup, not a cold or steady-state7000 tokens/s claim.

Strata's earlier no-draft controls gave78.8/79.4 TG with KV INT8. Those recording-inactive controls still used Nsight injection; our final native numbers have no injection. Strata also uses prepared HC BF16 weights versus our original Q8_0 HC weights. The remaining approximate10% TG gap is not a precisely isolated runtime difference, and parity is not established. The user's independently observed Strata TG80 is consistent with those controls. MTP remains deferred.

## Accepted mechanisms

1. Place Qwen HC RMSNorm and gamma multiplication on the actual layer device. This removes reverse gamma/activation transfers at layer boundaries without changing arithmetic. Eligibility: QWEN4EXP, layer split, multiple devices, full GPU layers, default context, one sequence, no recurrent snapshots or nextn embeddings. Existing backend support checks and fallback remain.
2. Reuse the canonical CUDA graph key from an existing entry with the same nonzero graph UID. Scheduler allocation gives a fresh UID when topology changes, and the existing graph-update path already trusts this immutable-UID contract. UID0, cache misses and eviction use the original full topology hash. No additional persistent cache, TTL change or capacity change.
3. Run independent routed/shared expert branches on two existing CUDA streams. Recognize the bounded Qwen `hc_mixed -> ffn_moe_out / ffn_shexp_gated -> ffn_out` cone; reject overlapping computed branches, external consumers, output nodes, CPY/SET_ROWS and reordered branch intervals. Existing allocator dependencies are retained for empty, prompt and decode shapes. Only single-row execution uses the events; post-allocation alias validation can fall back to sequential execution. Bound fusion to the current stream before the join.
4. Enable the existing fused MoE down/weight/sum kernel for Q8_0 K640/N2560,512 total/10 selected experts. All previous type, shape, contiguity and Ampere/Ada restrictions remain. Other extended combinations stay opt-in.
5. Group four independent Q8_0 gate/up rows per CTA, one row per warp, instead of the old two rows per warp/one warp per CTA. Guard: one query, K2560/N640,10 selected experts, fused SWIGLU, no separate gate destination or biases; existing Nvidia Ampere/Ada parent selector. Each output row keeps the same dot-product/reduction arithmetic.

No public API, attention budget, model files, KV precision, CPU sampling, global MMQ selectors or synchronization ownership changed. Nonmatching operations retain their existing paths. Graph-key reuse is backend-wide under the existing UID contract; the other changes have model/geometry guards.

Unset/nonzero enables the accepted controls:

```text
LLAMA_QWEN4EXP_HC_LOCAL
GGML_CUDA_GRAPH_KEY_REUSE
GGML_CUDA_QWEN_MOE_STREAMS
GGML_CUDA_QWEN_MOE_GROUP_ROWS
GGML_CUDA_QWEN_MOE_DOWN_REDUCE
```

Set an individual control to0 for diagnosis. `GGML_CUDA_MOE_DOWN_REDUCE=0` disables down fusion globally. `GGML_CUDA_MOE_DOWN_REDUCE_EXTENDED` remains OFF unless explicitly enabled; explicit EXTENDED=1 overrides the narrow Qwen exception. No experimental environment setting is needed for the final default launch.

## Frozen protocol

Weights: unchanged six-shard `/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf`, with `mmproj-F16.gguf` from the same directory. Architecture QWEN4EXP,48 target layers (36GDN/12QSA), hidden2560,HC4/low-rank320,512 experts/10 selected, routed/shared FFN640. GDN has48 value heads (not40). HC down/up/inject are Q8_0; norms F32. First shard is metadata-only; this is not a missing-weights condition.

Hardware: Threadripper PRO3945WX (12 cores/24 threads),188.5GiB RAM, five48GiB RTX4090/4090D and one24GiB RTX3090; CUDA13, Release, SM86-real/89-real. Default layer split9/9/9/9/9/3 in CUDA0/1/2/3/4/5 order, all layers/expert weights on GPU, pipeline enabled. PCI order:

| CUDA | GPU | UUID prefix | PCI prefix |
| --- | --- | --- | --- |
| 0 | RTX4090 | 9e5fb796 | 01 |
| 1 | RTX4090 | dd8b5c70 | 21 |
| 2 | RTX4090 | 3972c9ef | 22 |
| 3 | RTX4090D | 7f297c35 | 23 |
| 4 | RTX4090D | 8e5bef34 | 42 |
| 5 | RTX3090 | a84e3bf7 | 61 |

Server command (line breaks are shell continuations):

```sh
/home/sergei/Github/llama.cpp/build-aliceai/bin/llama-server \
  --model /home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  --mmproj /home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf \
  --ctx-size 262144 --batch-size 2048 --ubatch-size 512 --parallel 1 \
  --threads 20 --threads-batch 20 --flash-attn on --kv-unified \
  --fit on --fit-target 512 --cache-ram 32768 --load-mode mmap --lazy-mode auto \
  --jinja --image-min-tokens 1024 \
  --override-kv qwen4exp.attention.indexer.top_k=int:2048 \
  --no-warmup --no-webui --host 127.0.0.1 --port 18839
```

No drafter/MTP. F16 KV, CPU target sampling, text only (mmproj loaded). Raw `/completion` prompt IDs reproduce Strata `Service.encode_prompt` on the same saved public repository-document request, including its special-token escaping.21159 IDs, raw int32 SHA256 `19405fbe02a88bca7a0869f6ae909009b10fe078ebb1e9c073978e6d49bf39a7`. Payload: n_predict512, temperature1, top_p0.95, top_k20, min_p0.05, seed1234, cache_prompt=false, ignore_eos=true, return_tokens=true, stream=true, id_slot0. All requests report prompt_n21159/cache_n0. PP uses server prompt_per_second; TG uses server predicted_per_second (512 outputs,511 subsequent decode intervals). HTTP wall time is separately retained.

Each process: one warmup plus three measured requests. Baseline -> candidate without streams -> full candidate -> baseline; identical binary with individual flags0/1. In this pre-promotion binary Qwen down fusion was selected through explicit EXTENDED=1, all other flags explicit. Final promotion only changes defaults and narrows the down exception; additional guards restrict HC pinning to the tested context. Fresh default-on validation follows below.

Runners refuse occupied GPUs/ports, disable core dumps, watch all cards for at least512MiB free, and stop only their own process groups with SIGTERM, escalating on timeout. No user server was stopped. Nsight profiling is separate from clean native acceptance. File cache was warm; no disk-cold load claim.

## Correctness

| Gate | Result | Evidence under local archive |
| --- | --- | --- |
| Full-vocabulary logits after21159 PP and8 teacher decode rows | PASS, finite and byte-identical original reference | `group-rows/model-check.log`, `promotion/default-model-check.log` |
| Full sequence and recurrent-only states | PASS, byte-identical; full state769060020 bytes after PP,769306164 after8 decode | same logs and `moe-streams/reference/` |
| Cross-context restore then8 decode rows | PASS, full logits/state exact | same logs |
| Truncated-state failure/recovery | PASS, expected restore rejection/clear, successful restore/retry exact | same logs; error line is intentional |
| Clear,4096 PP after TG, decode, second-cycle state | PASS, exact | same logs |
| Synthetic M-RoPE embeddings | PASS, exact logits/state; not an actual image request | same logs |
| Graph key | PASS,2336 CPU key comparisons;216 CUDA computations/4.4MB exact on Ada/Ampere, UID0/fresh UID/eviction/context recreation | `graph-key/` helpers and operator logs |
| Fused down | PASS, existing CPU-reference MOE_DOWN_Q_REDUCTION tests on CUDA0/5 | `down-reduce/CUDA*-test-*.log` |
| Grouped gate/up | PASS,96 evaluations/737280 F32 outputs exact on CUDA0/5, dynamic inputs/IDs plus selected8, two-query and no-gate negative controls | `group-rows/operator-*.log` |
| Clean native generation | PASS, all16 responses x512 IDs match original baseline | `final-native/summary.json` |
| Default-on source/binary validation | PASS, original full-logit/state/lifecycle reference | `promotion/default-model-check.log` |
| Formal two-corpus PPL/NLL, real vision media, larger real contexts, multi-sequence, mixed CPU offload, MTP | NOT RUN in this experiment | No broader quality/mode certification |

Kernel activation confirmed in trace:24528 grouped-GU calls and24528 fused-down calls over511 target decode rows (48/layer). Operator timings alone were not used to accept the change. Existing sparse/state regression results are recorded in the final-validation supplement.

## Clean native acceptance

| Arm | PP request1/2/3 after warmup | TG request1/2/3 | First-request PP/TG |
| --- | --- | --- | --- |
| Baseline before | 4273/4212/4512 | 62.68/62.18/62.78 | 3679/62.08 |
| Candidate, streams OFF | 5699/6584/6967 | 68.75/69.10/69.59 | 4158/68.55 |
| Full candidate | 5598/6492/7035 | 71.34/71.38/71.69 | 4094/70.22 |
| Baseline after | 4279/4239/4527 | 62.82/61.93/62.68 | 3714/62.26 |

Six baseline observations median62.681, candidate median71.375 (+13.87%). Full request wall decreases from12.85-13.27s to10.16-10.97s. PP changes with CUDA graph-cache warmup/history despite disabled prompt reuse; compare request ordinals and preserve the first request. Do not describe the last PP sample as the cold/default startup rate or a guaranteed plateau. Full native manifests and raw SSE responses are retained.

## Diagnostic profile

Nsight2025.3.2, CUDA graph node tracing;511 decode rows. Profile overhead materially reduces TG. Sum includes overlap; union/idle are interval-union measurements, not SM occupancy.

| Variant | TG wall ms | Sum GPU kernel ms | GPU union ms | Overlap ms | No-kernel interval ms | Kernel count |
| --- | --- | --- | --- | --- | --- | --- |
| Original | 9084.844 | 7110.980 | 7110.124 | 0.856 | 1974.720 | 1684256 |
| HC placement + streams | 8706.450 | 7438.801 | 6870.187 | 568.614 | 1836.263 | 1684256 |
| + graph key | 8449.010 | 7440.541 | 6871.772 | 568.769 | 1577.238 | 1684256 |
| + fused down | 8453.960 | 7351.777 | 6783.803 | 567.974 | 1670.157 | 1610672 |
| + grouped gate/up | 8063.199 | 7261.955 | 6583.748 | 678.207 | 1479.451 | 1610672 |

Gate/up summed kernel cost2.509 ->2.141ms/token, near the separate Strata2.147 diagnostic. Fused down reduces launches; it is not a large isolated summed-matmul reduction. HC pinning removes15 H2D and15 D2H transfers/token. Cached graph-key reuse reduces measured pre-GraphLaunch host-gap median638.43 ->35.02us; most old hashing overlaps GPU execution, so this is not a3.5ms/token critical-path saving. The measured intersection with GPU-idle gaps is426.45 ->24.12ms over the trace.

A suspected72 recurrent-state copies/token are mostly captured graph nodes (36716 of36792 observed copies), not72 new host snapshots/token; no such copies were removed. Host sampling excluding sync is approximately0.317ms; no GPU sampling port was justified by this profile.

## Rejected or unproven variants

- Async input staging: native58.2-60.3 TG versus61.1-61.3; removed at user request, patch and measurements preserved under `rejected/async-inputs/`.
- Typed QSA metadata packing: decode-only changed reserve topology and regressed repeated PP; all-row dependencies restored PP but gave no clean TG gain. Removed; `packed-inputs/typed-metadata-all-rows.patch` and failed intermediate preserved.
- Fused GDN norm/weight/sigmoid gate: initial40-head guard was inactive (actual48). Corrected48-head branch and absolute output indexing were trace-confirmed120 calls and exact on operators/full model, but native gain was not established. Removed; both inactive and active evidence retained in `norm-gate/`.
- CPU threads20/12/4/1/20: final20-thread control matches1/4; no established thread gain. Default20 unchanged;128 teacher-logit rows and states exact in all five arms.
- Strata GPU stage order: no TG gain and different sampled IDs under changed placement; default placement unchanged. Separate four-GPU screen below is not a same-placement correctness control.

## Identity, evidence and reproduction

Pre-promotion acceptance diff SHA256 `e23a64179e0ac42afd46b156b64173b0d4a1932db751cba089f525b31c715243`, libggml-cuda `9ccb3b5161c3765a6fdcd9a981f427ae5a2fe863a78ded247525cfd9a450b0a8`, libllama `67e8b5e609d9d6afe7fef9332dcb5995d9fd1c9ec7c8b3c4485445651d01102b`.

Promoted source-only diff SHA256 `786b4080c0e80335315503ec5d156fa1575966084e4a3a2489f76cfc87794c6f`; libggml-cuda `543cb8837fe408b42dc8f18cc211c546120f84b8b0f9833ed80a7965ee8b0fd3`; libllama `d7108b32fc806f0c4d35a7eda563c7b457ea04e2a1a51bd7f37e3918fc2d7b68`. Server shim SHA256 `bd63ec65d6924e1d1945d3741b0f8308d2245f860976b4ccedce67b4fd6ac09e`; actual loaded modules, resolved paths, sizes and CMakeCache are in `promotion/source-binary-identity.json`. Source hashes exclude subsequent documentation additions.

Permanent local archive: `/home/sergei/_my_old/_my_glm53_flash/qwen38-strata-port-20261107/`. Working raw directory: `/tmp/qwen38-strata-port-20261107/`. Strata evidence: `/home/sergei/_my_old/_my_glm53_flash/strata-profile-20261107/`, binary SHA256 `e4406b474c292c8b52621c803f523ae18dc42c905deda60d5f055740c7ef8925`. Source/binary provenance, inputs, runner/helper sources, explicit environment/argv per arm, raw responses, memory and profiler SQLite/Nsight reports remain local. Only this REPORT.md belongs in Git, plus catalog/lessons/passport updates. No model rewrite, commit or merge performed.

## Final default-on validation and memory

Fresh rebuilt source with all experimental flags unset passes the original full logits/state/lifecycle reference. No new reference was generated for promotion. Clean native arms (one warmup plus three measured requests each):

| Arm | PP request1/2/3 | TG request1/2/3 | Peak all-card NVML GiB |
| --- | --- | --- | --- |
| Default six GPUs, before | 5774/7118/7736 | 71.42/72.00/71.86 | 141.645 |
| Optional four GPUs | 4332/5788/7169 | 72.43/72.85/72.86 | 139.125 |
| Six GPUs, down fusion OFF | 5479/6348/6968 | 69.55/70.14/70.55 | 141.459 |
| Default six GPUs, after | 5618/7149/7084 | 70.54/71.35/70.98 | 141.574 |

All12 six-GPU responses including warmups match the original baseline512 IDs. Default TG median across both controls71.387, down-OFF median70.144; down fusion is retained. First-request PP/TG: default-before4130/70.84, default-after4072/70.61, four-GPU3417/66.58. The higher repeated PP numbers retain the same graph-cache/history caveat as earlier acceptance.

Four-GPU command appends `--device CUDA0,CUDA1,CUDA2,CUDA4 --tensor-split 1,1,1,1`. This is an optional placement screen, not a default/config change or an equivalent same-output speed comparison: all four sampled512-token arrays differ from the six-GPU reference starting at index7. It frees GPUs3/5 but first-request PP is lower. Full-logit cross-placement quality was not measured; do not infer its modest measured TG advantage is solely transport savings.

Memory watcher samples every5s; numbers are simultaneous maximum NVML used across all cards, including the small driver baseline, not just weight bytes or a microsecond-resolution allocation maximum. Six-GPU peak per card MiB:27493/26631/26619/27219/26583/10499 (sum145044MiB =141.645GiB). Four-GPU peak:38763/35251/35261/18/33153/18 (sum142464MiB =139.125GiB). Intermediate allocations can be smaller before the first full PP/TG cycle; the125.5GiB snapshot was not the full-request peak. Sampled minimum free memory in the default runs13626/13632MiB, four-GPU9746MiB, all safely above the512MiB floor. This is ctx262144 F16 KV, mmproj loaded, no draft.

Existing `test-glm5next-sparse` and `test-save-load-state -m build-aliceai/tests/test-models/qwen4exp-moe.gguf -ctk q8_0 -ctv q8_0 -n 4` pass on the rebuilt default binary. The state test's incompatible-rotation restore error is an expected negative case. Exit manifests are in `promotion/checks/`; all own native/server/check processes exited0 with no watcher stops, GPU process query is empty after cleanup. Source diff whitespace check passes. No rejected async/packing/norm-gate implementation remains in the working sources.
