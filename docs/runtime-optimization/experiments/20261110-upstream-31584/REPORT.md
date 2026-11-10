# Upstream #31584: strided view bounds, 2026-11-10

Status: exact ggml.c port applied and targeted gates pass; committed at the user's request. Branch my_build, base4da32d30b. The preceding #31350/#31368 core port was committed as4da32d30b at the user's request; unfinished GLM work remains outside that commit.

## Change and local adaptation

Upstream bbb02c93e9b83255b70ac1a755bb02b0a251f9a7 checks a view against its root source using the bytes reached by the final strides. Previously the check used the contiguous size of the shape before the supplied strides were assigned. That could reject overlapping rows that fit, while accepting a wider stride that reaches beyond the buffer.

The internal tensor constructor now receives strides before the check. View2D/3D/4D pass their complete strides, view_tensor preserves the source layout at construction, and strided views require contiguous elements along source dim0. Other constructor callers pass NULL strides. Public interfaces, kernels, weights, state formats and runtime defaults are unchanged. Root-view offset flattening and empty-view behavior are preserved.

Only ggml/src/ggml.c changes in the repository. The upstream delta-net-base.cpp hunk removes a workaround that widens a one-window view after checking it. Our current convolution snapshot implementation creates separate bounded3D views and has no such widening; that hunk does not apply. Its existing GLM scalar-window edits remain byte-identical. No convolution snapshot packing or arithmetic rewrite was added.

## Gates

| Gate | Result |
| --- | --- |
| Baseline/candidate constructor oracle | All2020 candidate cases match the independent byte-span oracle |
| Baseline failure reproduction |404 valid cases rejected by baseline;185 invalid cases accepted by baseline |
| Ordinary accepted layouts |1137 baseline/candidate outputs have identical shape, strides, span, root and offset metadata |
| Actual view -> clone -> CONT |72 byte-exact CPU copies: overlap, gaps and zero stride; F32/F16/BF16/Q8_0/Q2_K/Q3_K/Q4_K/MXFP8 |
| Existing backend CONT suite, CPU |87/87 pass, including DeepSeek indexer permutation shapes |
| Private compiler gate |llama-server, test-llama-archs, test-recurrent-state-rollback and test-backend-ops build |
| CPU architecture/state |DeepSeek4, DeepSeek41, GLM5Next and Qwen4Exp pass; Qwen3Next passes as an RS-disabled control |
| Existing recurrent rollback executable, CPU |All four saved tiny GGUF fixtures pass; none is skipped as non-recurrent |
| CUDA architecture/state |Four architectures pass on each Ada/Ampere device with the unchanged installed CUDA module |
| Actual two-GPU layer split |DeepSeek41 and GLM5Next pass; first/last trunk devices are CUDA0/CUDA1 |
| Unfinished GLM manual windows, CPU |Scalar and token executor each pass56 window cases,10 controlled faults/retries and5 fallbacks with exact logits/state |
| Source preservation |All21 starting modified/untracked files retain their SHA256; normalized final compiler commands match baseline |

Constructor cases cover allocated and no_alloc contexts, dimensions1..4, exact end and one-block overflow, overlapping/broadcast/wide strides, empty shapes, nested offsets, view_tensor on a transposed source, invalid strided access to a transposed dim0, and reproducible random layouts. Types are F32/F16/Q8_0/Q2_K/Q3_K/Q4_K/MXFP8. The copy fixture also covers BF16 and compares physical source blocks against every output row. It does not dequantize or measure speed.

## Initial build mismatch and causal control

The first fresh CMake configuration enabled AVX/AVX2/FMA/F16C/SSE42/BMI2 by default. The frozen baseline had those disabled because of its earlier configuration history. Merely setting GGML_NATIVE=OFF did not make the two configurations identical. The initial vector build passed architecture and GPU gates but failed the GLM multi-sequence recurrent bit-exact oracle with a5.82077e-10 difference in cache_r_l1; the same failure repeated5/5 times. No tolerance or runtime logic was changed to hide it.

A crossed control loaded each CPU backend with both core libraries:

| Core | CPU backend | GLM bit-exact recurrent fixture |
| --- | --- | --- |
| Before #31584 | Baseline scalar | PASS |
| Before #31584 | Initial vector | Same5.82077e-10 failure |
| After #31584 | Baseline scalar | PASS |
| After #31584 | Initial vector | Same5.82077e-10 failure |

This local control attributes that failure to the CPU build path, independently of this view-bounds patch. It is not evidence of a #31584 numerical regression, nor a general certification of vectorized recurrent exactness. Final candidate CPU options were explicitly matched to baseline, normalized compiler commands were checked, and all CPU model/checkpoint/manual-window gates and GPU architecture gates were rerun successfully. Initial vector binaries, failures and controls remain in the local evidence. These private test options do not modify repository defaults or the installed runtime.

## Isolation, evidence and limits

Work directory: /tmp/dsv41-upstream-31584-20261110. Local archive: /home/sergei/_my_old/_my_deepseek41/upstream-31584-20261110. The archive records upstream/source patches, source hashes, pre-existing file hashes, private binaries, commands, models' hashes, constructor/copy fixtures, CPU build controls and complete logs. Generated tiny models and startup/failure helpers are reused from the recorded preceding rollback fixture; the archive includes the helpers for remapping/reproduction. Only REPORT.md is intended for Git.

GPU UUIDs: GPU-9e5fb796-93b1-b72d-1fd8-438d93aac573 and GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783, CUDA_DEVICE_ORDER=PCI_BUS_ID. The old CUDA permutation is not used. Actual GPU tests were run only after a free-GPU check. No installed library, model or preset was replaced. This is a metadata/lifecycle regression screen, not a full trained-model performance or quality comparison.

Full Q3 with DSpark K3/K5, long-draft HTTP acceptance, text/code NLL/PPL, max-context memory and native PP/TG remain untested in this stage. No speed gain or fix for the historical floating6 token10 divergence is claimed. These gates remain necessary before deployment. Next in the upstream queue: separately review #30662 graph idle eviction and #31679 compiled-architecture MMVQ selection, preserving our scheduler/cache/retained-plan guards.
