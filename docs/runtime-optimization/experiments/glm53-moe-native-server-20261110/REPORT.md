# GLM rank-first MoE: native DFlash server follow-up

Date: 2026-11-10. Branch `my_glm53_flash_1`, HEAD `c8000c1fa97c21705eeb6f3668f85724c145d0fe`, existing dirty stack retained. Status: native measured opt-in candidate, default OFF. No code changes, commit, deployment or default promotion in this follow-up.

## Result

| Arm | PP tokens/s | TG tokens/s | Request wall s | Min free VRAM MiB | Min available RAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| off1 | 1807.184 | 61.686 | 36.064 | 1104 | 169.989 |
| on | 1802.340 | 63.249 | 35.868 | 1106 | 169.810 |
| off2 | 1807.735 | 61.703 | 35.997 | 1106 | 170.158 |

Medians of three 50000+512 requests per fresh process. ON TG versus OFF1 +2.533%, versus OFF2 +2.504%; pooled-OFF-relative TG +2.519%, PP -0.283%, whole request wall -0.451%. OFF2/OFF1 TG drift +0.028%. Baseline IDs/counters repeatable: True. All-arm IDs/counters exact: True. Predeclared screen passed: True.

Gate declared before execution: all short/long IDs and aggregate counters exact, ON median TG >=1% above each OFF median and PP loss <=1.5% against pooled OFF. Every row including cold controls is retained, with no retry/selection on agreement. One ON startup is a bounded bracket, not general statistical certainty or broad quality certification. Do not add gains from earlier cohorts.

## Mechanism and prior correctness

The [real-route experiment](../glm53-moe-real-inputs-20261110/REPORT.md) documents the narrow source implementation: only Ada CC89, Q5_K fused up/gate, K4096/M2048/N5/top8, no auxiliary gate output/bias/scales, eligible GLU. Grid1024x8 becomes8x1024 with the same block32x5 and per-row dot/reduction/fusion order. Other geometries/architectures keep their original path. `GGML_CUDA_MOE_RANK_FIRST=1` before server startup enables it; unset/0 stays OFF.

Prior evidence remains separate: actual route capture, corrected64-case real-weight/input operator controls, exact target logits/features/state/rollback and original-frozen reference, balanced OFF/ON/sham target-window timings. Native source-port operator fixture passes64 exact comparisons on Ada/Ampere, confirms candidate/fallback grids and uses no math override. Those metadata/precision observers are absent from this server cohort. No fresh full tensor introspection, raw logits/state, per-window proposal/acceptance lattice or NLL corpus is collected here; aggregate API equality is the new consumer check. Earlier cross-restart distribution variability is not repaired by this matching cohort.

## Frozen native protocol

This is a real source-integrated CUDA backend with no LD_PRELOAD, profiler, evaluation callback or dispatch/readback observer. Same binary bytes in all three arms; only `GGML_CUDA_MOE_RANK_FIRST` changes unset/1/unset, verified from process environments. Process module maps/hashes verify the native backend, identical runtime libraries and absence of previous rank/metadata observer modules. Source stays unchanged throughout execution.

Local evidence root E: `/home/sergei/_my_old/_my_glm53_flash/glm53-moe-real-inputs-20261110`. Runtime `E/native-runtime`, results `E/native-server-01`. CUDA SHA256 `719b0e3ef8d288cbd4b48f4abc69d1858cf2866be2285bc63a8dfa440fcf1b20`; server `4a9fb6ab67d8ad6eba5b617b160b68a33c8b2fcccc338ea1d932178c412c667b`; libllama `b7b70db1729c060d8a32daf9641d0f1b10783b020ee3218de5c1a24e3012b501`. The previous original-object relink reproduced the frozen CUDA hash exactly; the native build replaces only the mmvq object. Existing shared-build and installed binaries are untouched.

Unchanged unsloth six-shard GLM-5.3-Flash UD-Q5_K_XL plus Anbeeld GLM-5.3-Flash-DFlash2-Q5_K_M/max4, CPU mmproj loaded/text only. CUDA PCI_BUS_ID order, five48GiB Ada plus one24GiB Ampere. Context278528, B8192/U1024, parallel1, threads20, target/draft unified F16 KV, FA ON, model indexer top-k2048. Fit OFF, load none, historical24 overrides/split11,8,8,8,8,4, all experts GPU. Exact argument/rule identity and unchanged model/placement source inherit prior mapped1383 target/81 draft tensors; no fresh native tensor-map observer is claimed.

Cache RAM0/cache_prompt false; seed1234, temperature0.95/top-p1/top-k40/min-p0.05, CPU target sampling, ignoreEOS, frozen raw prompt IDs, streaming and returned IDs. Prompt SHA256 `29bac06346b966452bf31ebbc4dde8114eb79e55a1bb7806f16aac3d933172e6`. Standard protected prefill, GPU override permission and DFlash GPU features ON; mHC split/scalar/two-stage/sparse-reset experiments OFF. Three fresh OFF/ON/OFF processes, each three2048+64 controls then three50000+512 requests. Full prompt/output counts and cache_n0 verified. Common startup probes/warmup remain, OS caches/clocks are not reset or locked.

## Resource and lifecycle gates

Prior paired no_alloc GGML plans have zero delta; they do not measure lazy CUDA scratch. Fresh OFF1 sampled free>=768MiB is required before ON. Every real process has live512MiB/GPU and4GiB available RAM floors, foreign-owner guard, bounded startup/HTTP waits and cleanup of its own process group. Sampled minima are in the table; they are not instantaneous peak measurements. All three processes exit0, no watcher faults, owners empty after each arm. No hardware performance counters, sudo or driver/security/clock changes.

Reproduce analysis with `E/analyze-native-server.py`; protocol `native-server-protocol.json`, exact launch/controller/source/binary identity, API chunks, metrics, memory/module/environment logs and final manifest are local. The runner needs free GPUs and retains all rows. `native-server-analysis.json` contains aggregate checks and unrounded medians; per-request metrics and IDs are retained in each arm's `results.json`.

## Disposition

Keep the source flag default OFF. This follow-up passes the declared native-server screen: matched IDs/counters, >=1% TG above both controls and no material PP loss. It does not authorize default promotion/installation/commit. Both arms use DFlash: no DFlash-versus-no-draft speedup, MTP result, ordinary N1 gain, media/other-model/architecture admission or Maya gap closure is inferred. No cache-hit/DRAM causal claim. Wider N2/N3/N4 or other devices require independent measurement rather than broadening this guard from N5 evidence.


## Subsequent default ON promotion (2026-11-10)

The user explicitly requested default ON and a separate commit after the completed native screen. Unset `GGML_CUDA_MOE_RANK_FIRST` now selects the same eligible launch as explicit1; explicit0 keeps the original launch, as do other parsed numeric values. Set overrides before server startup. Ada/shape/type/fusion guards, arithmetic, grids, placement and buffers remain unchanged. All earlier default-OFF protocol/results above describe the measured cohort and are not reclassified as a default-ON benchmark. This policy change adds no MTP/media/NLL or general restart-repeatability coverage. Installed binaries are not replaced.
