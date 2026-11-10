# Upstream #30662 and #31679: CUDA graph lifetime and MMVQ dispatch, 2026-11-10

Status: adapted source and targeted gates pass; committed at the user's request. Branch my_build, base1a66bbbad. The preceding #31584 port and its report were committed as1a66bbbad at the user's request. Installed libraries, models and presets were not replaced; unrelated unfinished GLM work remains outside these ports.

## Changes

### #30662: prevent premature graph eviction

Upstream00f113eceb2758158582d56f98bf476b2fe3ed32 excludes the requested graph from an age-based eviction sweep and gives cached graphs a reuse window after an idle period. Our adaptation keeps structural uint64 keys, hard capacity, configurable TTL, graph statistics and existing destruction/stream ownership.

The context records next_graph_eviction_time and last_graph_use_time. If a nonempty cache resumes after at least its configured TTL, its next age sweep is delayed by one TTL. Active sweeps remain5s apart; the requested key is never age-evicted by that lookup. A miss still evicts the least-recently-used entry immediately when capacity is exhausted. Grace does not bypass the capacity limit or refresh every entry's last-used timestamp.

The upstream fixed10s idle threshold/grace is expressed through our existing TTL setting. Default capacity64 and TTL10000ms, accepted capacity1..512 and TTL1..60000ms, fallback parsing, counters and disabled-stats behavior remain. The optional timestamp argument is internal and allows deterministic component checks; production callers still use ggml_time_us(). No scheduler events, graph keys, copy slots, retained-plan ownership or kernel arithmetic changes.

### #31679: use the architecture actually compiled into MMVQ

Upstream016c1d914c6e110318d20d2a8dbc2037a77e95f7 adds ggml_cuda_highest_compiled_arch(cc) inside the NVIDIA branch of get_mmvq_mmid_max_batch. Our port is exactly that one line. It preserves our existing Volta/Pascal behavior, AMD tables, Q3 Ampere maximum5 and the existing6-query5+1 fallback. Earlier unfinished GGML_CUDA_MMVQ_SPLIT_COLS code is unchanged.

On Ada with an SM86-only build, the old host selector returns8 for Q3_K while the compiled device limit is5. The candidate returns5. On our native SM86+SM89 configuration, actual Ada still returns8 and actual Ampere still returns5; this port is a dispatch consistency fix, not evidence of a speed gain on these native cards.

## Validation

| Gate | Result |
| --- | --- |
| Full private CUDA builds |Baseline and candidate compile every backend TU with SM86-real/SM89-real, graphs and NCCL enabled; no mixed context layouts |
| Cache component fixture |46 candidate cases pass on Ada/Ampere using the production cache method and real CUDA Graph executables |
| Baseline negative controls |Current-key loss at active sweeps and idle reuse failures are reproduced; ordinary capacity/teardown controls pass |
| Capacity/TTL/stats |Default, capacity1/3/512, TTL1/1000/60000ms, invalid/zero/empty/oversized settings and stats disabled; every reported teardown has zero live graphs |
| Compiled-SM selector matrix |SM75-only, SM86-only and SM86+89 binaries:3080 candidate host-table comparisons and264 actual device-type comparisons pass on Ada/Ampere |
| Selector negative control |Old Ada SM75-only/SM86-only selectors differ from6 device type limits per build; candidate has zero differences |
| Native routed operators |Existing test-backend-ops --test-file runs64 cases on each card for each variant; all pass CPU-reference tolerance |
| Architecture/state regressions |DeepSeek4, DeepSeek41, GLM5Next and Qwen4Exp pass on each card in both variants; explicit mixed Ada/Ampere split passes DeepSeek41 and GLM5Next |
| Cold/warm/idle HTTP |Seven requests per variant with11.1s idle; all112 generated IDs are identical across baseline/candidate |
| Preservation |All21 starting changes preserved:20 files byte-identical, dirty MMVQ file differs only by the one-line port |

The cache fixture uses an executable-local clock alias in the included production header, actual CUDA Graph memset nodes, output checks, retained uid/warmup metadata and production context destruction. It checks active expiration, current-key protection, grace ending, hard capacity, hit/create/eviction accounting and stats-off behavior. The baseline lacks the new idle policy and correctly fails those reuse or grace-deadline oracles; the candidate must pass every case. This component control does not simulate GPU completion by changing runtime events.

Selector fixtures extract the unmodified production architecture tables, host selector and device constexpr selector, then compile them for the listed SM sets. Device kernels report their compiled limits. The matrix covers all44 type enum values and NVIDIA/AMD host-table cases where the compiled NVIDIA floor exists; it is not an AMD/HIP runtime test. Deliberately invalid over-limit kernels are not launched. The full native backend is also built and exercised independently.

The64 routed cases use Q8_0/Q2_K/Q3_K/Q4_K, m64/512, k256, four experts/two selected, N1/2/4/5/6/7/8/9. They use the existing generic-op input format and its CPU-reference tolerance5e-4; random inputs are generated by that existing harness, so these are per-run reference checks, not cross-variant bitwise comparisons. No repository test file was added. Architecture fixtures also exercise same-backend rollback/state/restore checks; they do not establish trained-model quality or cross-variant bitwise raw logits.

## Native HTTP graph counters

The small DeepSeek41 fixture uses the existing synthetic weights with a minimal tokenizer added to a separate local GGUF. A C++ comparison confirms all99 weight tensors unchanged. There are five pre-idle requests and two post-idle requests, fixed input IDs, greedy sampling,16 generated tokens per request, cache_prompt=false, B64/U16, one GPU. Baseline and candidate use the same host libraries and model; only the complete CUDA module changes. The server caps the requested512 context to the fixture's trained256. Verbosity5 exposes the existing graph statistics; no eval callback or profiler is installed.

| Counter, entire seven-request server lifetime | Baseline | Candidate |
| --- | ---: | ---: |
| Compute calls |640|640|
| Cache creates |72|47|
| Cache hits |3323|3363|
| Age/capacity evictions |47|0|
| Peak entries |47|47|
| Captures |40|30|
| Instantiates |40|30|
| Launches |585|610|
| Update failures |0|0|
| Entries after teardown |0|0|

This confirms reduced reconstruction after idle in this workload. It does not measure a steady-state TG gain, a full-model TTFT gain or the memory cost of retaining trained-model graphs. Hard capacity remains unchanged; grace can retain more entries after idle than the old sweep, within that limit.

## Setup failures and evidence

Initial local fixture attempts are kept or documented: an nvcc linker-option spelling and a missing standalone CUDA error helper; an old-cache negative oracle initially recognized only lost-graph failures and needed to recognize its different5s sweep behavior under a short TTL. The original architecture GGUF has tokenizer type none and cannot be used directly through server sampling. An initial GGUF helper tried to reuse loader-only tensor-info state for writing; a new writer context with proper loaded tensor handles fixed that helper. The copied fixture's weights were compared afterward. An optional Python reader lacked numpy, so the weight check used the C++ GGUF API without installing anything. Server default verbosity suppressed INFO graph counters; the final comparison uses verbosity5 on both variants. None of these are claimed as runtime repairs.

Work directory: /tmp/dsv41-upstream-graphs-20261110. Local archive: /home/sergei/_my_old/_my_deepseek41/upstream-graphs-20261110. It records upstream and port-only diffs, before/after sources, pre-existing working-tree snapshots, module hashes, private builds, cache/selector/operator/HTTP fixtures, source/model identities, commands and logs. Only REPORT.md is intended for Git.

GPU UUIDs are GPU-9e5fb796-93b1-b72d-1fd8-438d93aac573 (Ada) and GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783 (Ampere), CUDA_DEVICE_ORDER=PCI_BUS_ID. GPUs were checked free before the owned tests; the obsolete device permutation is not used. All owned server/test processes exited.

Full trained Q3+DSpark K3/K5, long-draft HTTP acceptance, NLL/PPL, max-context512MiB memory-floor admission and native PP/TG were not run. The historical floating6 divergence and separate experimental GLM retained/token-executor questions remain outside these fixes. Private host libraries retain the prior unfinished GLM hooks, without promoting or committing those prototypes. Installed runtime is unchanged. Next queue items are the separately scoped #30553 StreamK/#31247 Q8_1 experiments and #31321 loader work; their arithmetic/resource gates must not be inferred from this result.
