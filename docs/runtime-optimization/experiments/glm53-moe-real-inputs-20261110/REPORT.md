# GLM-5.3-Flash: real MoE routes and rank-first fused up/gate grid

Date: 2026-11-10. Base: `c8000c1fa97c21705eeb6f3668f85724c145d0fe`, branch `my_glm53_flash_1`, existing dirty pipeline/input/mHC stack retained. Status: measured candidate, source integrated opt-in/default OFF; no commit, production installation or default promotion.

## Result and mechanism

Ada fused Q5_K up/gate at K4096/M2048/N5/top8 uses grid `[8,1024,1]` instead of `[1024,8,1]`, retaining block `[32,5,1]`. Row tiles and expert-rank axes exchange roles; input/output coordinates, per-row dot/reduction order, fusion and rounding remain unchanged. No new buffers, streams or scheduler ownership. The hypothesis is better expert-weight locality across tokens whose matching experts occupy different top-k ranks. CUDA block execution order is not guaranteed; no cache-hit/DRAM counters or exclusive cache-cause proof is available.

Eligibility is deliberately narrow: Ada CC89/warp32, Q5_K, K4096/M2048/N5/top8, fused gate, no auxiliary gate output/bias/scales, SWIGLU or SWIGLU_CLAMP. All other cases retain their previous launch. The integrated flag is `GGML_CUDA_MOE_RANK_FIRST=1`; unset/0 and other values use the old launch. Set it before context construction/capture. This changes speculative N5 verification, not ordinary N1 decoding or prefill dispatch.

Actual DFlash server medians (three long requests per fresh process):

| Arm | PP tokens/s | TG tokens/s | Request wall s |
| --- | ---: | ---: | ---: |
| OFF1 | 1803.110 | 61.396 | 36.145 |
| ON | 1802.060 | 63.279 | 35.857 |
| OFF2 | 1805.544 | 61.825 | 36.027 |

ON TG is +3.066% versus OFF1 and +2.352% versus OFF2; pooled-OFF-relative +2.876%. Baseline drift is +0.697%. PP change is -0.126%, request wall -0.635%. All short/long output IDs and aggregate draft counters match across all arms. These performance measurements use the same external recompiled CUDA module plus metadata observers in every arm, not a clean native shipping-binary cohort. Native source-port correctness and physical activation are checked separately below. One ON startup is a bounded screen, not general statistical or production admission.

## Frozen protocol and identity

Local evidence root: `/home/sergei/_my_old/_my_glm53_flash/glm53-moe-real-inputs-20261110` (`E`). Sources, expanded argv/environment, IDs, binary/source hashes, raw inputs/results, build logs, analyzers and `SHA256SUMS` remain there. No raw weights are added to Git.

Measured frozen runtime: `../glm53-mhc-dflash-server-20261109/runtime` relative to E. SHA256: server `4a9fb6ab67d8ad6eba5b617b160b68a33c8b2fcccc338ea1d932178c412c667b`; libllama `b7b70db1729c060d8a32daf9641d0f1b10783b020ee3218de5c1a24e3012b501`; libggml-cuda `5eb7e080641a89b9397f5a9bc8276b8bc578cb17385fbe8c7c37f097f8aa39cf`. Five 48GiB Ada GPUs and one 24GiB Ampere GPU, CUDA PCI_BUS_ID ordering, exact UUID/PCI/maps in the archive.

Target: unchanged six-shard unsloth GLM-5.3-Flash UD-Q5_K_XL; draft: Anbeeld GLM-5.3-Flash-DFlash2-Q5_K_M. Actual server: ctx278528, B8192/U1024, parallel1, threads20, unified F16 target/draft KV, FA ON, indexer top-k2048, all experts on GPU, fixed split11/8/8/8/8/4 plus identical24 overrides. Text only; CPU mmproj loaded. Fit OFF, load none, cache RAM0, cache_prompt false. Draft max4/min0/pmin0, all draft layers on GPU; CPU sampling temp0.95/top-p1/top-k40/min-p0.05/seed1234/ignoreEOS, raw frozen IDs and returned token arrays. Standard protected prefill and GPU feature path ON; mHC/scalar/two-stage/reset experiments OFF. Exact argv is in `server-01` and `server-protocol.json`.

Three fresh server processes OFF/ON/OFF each run three 2048+64 and three 50000+512 requests, 18 total. Same modules in all arms: `rank-ada.so`, startup metadata observer, cold graph metadata observer. Prototype flag `GLM_MOE_RANK_FIRST=0/1/0`; only the later source integration renames it. No routing readbacks, eval callback, hardware counter profiler or extra head override during server timing. Actual1383 target/81 draft tensor records, types, bytes and placement match. Paired no_alloc plans have zero difference; this does not cover lazy CUDA workspace. Fresh-baseline margin768MiB, live floor512MiB/GPU and4GiB available RAM, foreign-owner guard, owned-process cleanup. All three processes exit0, no guard fault, minimum sampled free1106MiB and available RAM170.096GiB; GPUs released.

## Route capture and operator controls

`full-01`: one target-only ctx65536 load with50K checkpoint and32 consecutive forced N5 appends. Preallocated32MiB pinned staging, stream-ordered D2H capture, drain/flush and poisoned-buffer checks; profiling run separated from timing. Capture includes all42 routed up layers plus two down layers per window:1408 records. Before/after160-token logits, features and full state are byte-exact. Final state SHA256 `4a44f71b0b177b28045fa9959390318f04225120d2f365c152c5b93f2f39671e`.

`analyze-capture.py`: among40 selected IDs/window/layer, unique experts median29/mean27.6488; repeated selection fraction30.878%. Of cross-token same-expert pairs,78.935% occupy different ranks. Logical unique fused-Q5 up/gate footprint median319MiB versus Ada L2 capacity72MiB (capacity query only, not measured traffic). These observations motivate the grid candidate; they do not prove a cache mechanism.

Real weights/inputs/routes at eight positions50000..50140, four operators (Q5/Q6 up, Q6/Q8 down), Ada+Ampere:64 exact output comparisons. Captured ID stride288 int32 retained, GLU op6/SWIGLU_CLAMP limit10 reconstructed. Six alternating rounds,16 repeats, warm replay and preceding large-memory-sweep controls; sweep excluded from timing.

Initial shared first-node graph anchors collided in CUDA graph keys and caused recapture on policy changes. Preserve `operator-initial-shared-anchor` as ineligible timing evidence. Corrected `operator-02` uses distinct policy anchors: exactly8 cold captures/device, no switch recapture. Ada Q5 median warm474.51->454.89us (-4.13%), swept522.33->502.09us (-3.88%); all eight input frames improve. Both kernels use53 registers,0 local bytes and theoretical7 active blocks (not measured occupancy). Unchanged Ada controls remain near zero. Ampere unchanged controls still show large timing differences, so Ampere performance is inconclusive and excluded from source eligibility. Exactness passes on both architectures. `operator-final-checks.json` records the predeclared Ada >=3% operator gate and subsequent Ada-only/sham refinement.

## Full target and actual draft gates

`append-01`: Ada-only module plus sham mode2 (old kernel), identical ctx65536/50K checkpoint, six balanced three-policy orders,32 warm+16 timed N5 windows per arm.288 timed windows; no route D2H. OFF58.0020ms, ON56.2694ms, sham57.9765ms. ON cost -2.965% versus pooled OFF/sham, six paired medians -2.576..-3.235%; sham/OFF -0.044%. Physical grid metadata confirms Ada-only activation, Ampere fallback. Predeclared >=1% full-window improvement/sham within0.5% passes.

Cold/warm N2/N5 logits/features/state, keep1/3/4 rollback followed by N1, and final240-token state/features/logits are exact across all policies. Raw N2/N5 files and state hashes also match the prior original frozen-runtime reference, not just the recompiled OFF module. This forced target diagnostic is not accepted-token server TG. Minimum free3696MiB/available RAM166.666GiB, exit0, guards clear.

Actual server outputs repeat within/across all three starts: short SHA256 `65527e3c6b36bbd22ef618dac6587c09ceb7521d57bf89913c7586f4362c0c32`, long `5a206d0aa7dd0ceed1145dc24838cc4c3112a5a13fdd9328c79064da28bdaace`; short aggregate draft counters66/46, long479/391. Six cold selection records per startup confirm the candidate Ada grid and unchanged Ampere launch. No per-window proposal/lattice/raw-state trace is collected in this server cohort; aggregate equality is bounded evidence. Earlier cross-restart probability variability remains unresolved and is not repaired by this matching cohort.

## Source integration and native validation

After all measured phases, only `ggml/src/ggml-cuda/mmvq.cu` gains this candidate on top of its existing uncommitted mHC changes. `native-integration.patch/json` separates the new delta. Compared with the measured module: environment name, named Ada constant, comment and argument spacing change; device math and admission remain equivalent. Source hashes and all unchanged dirty-stack/runtime hashes are retained in `identity.json` and final checks.

The isolated `native-runtime` copies the frozen runtime and replaces only its CUDA library. Existing original CMake object relink reproduces frozen CUDA SHA256 byte-for-byte (`native-original-relink.json`); the new link changes only the freshly compiled mmvq object. Release/use_fast_math, SM86/89, original linker inputs; no installed or shared build library is overwritten.

Native build completes successfully (85.244s compile); CUDA library SHA256 `719b0e3ef8d288cbd4b48f4abc69d1858cf2866be2285bc63a8dfa440fcf1b20`. All64 real-input comparisons pass, with8 initial graph captures per device and correct Ada/Ampere grids. Minimum sampled free19522MiB and available RAM174.989GiB; both processes exit0/guards clear/GPU owners empty. The copied server `--version` exits0. Native check results are recorded in `native-checks.json` and `native-check-01`: real captured inputs/weights, all four operator families on Ada/Ampere, byte-exact OFF/ON and finite outputs, physical activation/fallback, loaded native CUDA maps. No math override is preloaded; only a cold metadata observer. These are correctness/dispatch checks, not new native server speed measurements. The copied server executable is unchanged; use the adjacent new CUDA library through the isolated runtime. Default remains OFF.

## Decision and remaining scope

Keep the narrow source candidate opt-in/default OFF. Server TG improvement is repeatable against both bracket controls in this cohort and needs a clean native server bracket before production/default promotion. Do not combine it with unrelated placement/head/two-stage candidates or sum independent percentages. No ordinary N1 speedup, PP improvement, MTP/media/other-model quality, Blackwell/HIP support or Maya speed-gap closure is claimed. Failures/cancellation and general NLL/corpus admission are not established here. No commit or installed-binary replacement.

Reproduce analysis with `analyze-capture.py`, `analyze-operator.py`, `analyze-append.py`, `analyze-server.py`; read each phase's protocol before execution. GPU runners require free devices and retain resource guards. `native-build-argv.json` and `native-fixture-build-argv.json` preserve exact build commands. Source activation: `GGML_CUDA_MOE_RANK_FIRST=1`; unset or0 keeps the original path.


## Subsequent native server follow-up (2026-11-10)

The [separate native-server report](../glm53-moe-native-server-20261110/REPORT.md) closes the no-preload measurement gap for this bounded workload: TG61.686/63.249/61.703 in unset/ON/unset, all IDs/counters exact, declared screen passes. This does not rewrite the external-module measurements above or promote the default. Source remains opt-in; no installation or commit.
