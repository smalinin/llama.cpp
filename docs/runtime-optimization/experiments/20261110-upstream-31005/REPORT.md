# Upstream #31005: post-bias NaN sanitation, 2026-11-10

Status: port applied and targeted CUDA gates pass; recorded by the user-requested commit. Branch my_build, base a71f944b3. User requested starting the upstream fix queue after the GLM-5.3-Flash additions. Existing GLM, scheduler and documentation changes are preserved. No broad upstream merge, model benchmark or installed runtime replacement was performed.

## Change

Exact three-line change from upstream commit 10d4846a572aebf21c0482d140b00e74c0f1ad20, ggml-cuda: sanitize post-bias NaNs in MoE selection (#31005). In ggml/src/ggml-cuda/topk-moe.cu, a NaN selection score after adding expert bias becomes -FLT_MAX. The earlier sanitation of routing weights cannot catch a NaN introduced by this addition. Ordinary scores retain their existing arithmetic. Padding and the selected-expert -INFINITY sentinel are unchanged.

No GLM rank-first grids, MMQ/MMVQ selectors, graph cache, scheduler, sampling or DSpark state code changed in this stage. This is a correctness guard, not an established explanation of the historical floating6 mismatch or an inference speed optimization.

## Validation

| Gate | Result |
| --- | --- |
| Exact upstream patch | git apply --check passes; source diff has only the three upstream additions |
| Production CUDA translation unit | Compiles with the recorded build-aliceai flags, SM86 and SM89, exit0; output object isolated from the installed build |
| RTX4090, SM89 | 2640 operator cases pass |
| RTX3090, SM86 | 2640 operator cases pass |
| Ordinary finite inputs and ties | 1280 cases per GPU have bitwise-identical old/new weights and IDs, including padding/guards |
| Exceptional biased selection | 1360 cases per GPU match CPU selection/weight checks; selected IDs follow the reference without duplicates; weights are finite for these fixtures |
| Negative control | Old/new IDs differ in1040 exceptional cases per GPU; the candidate matches the CPU reference, reproducing the old failure |
| Existing work preservation | SHA256 unchanged for all21 previously modified/untracked files captured before applying this patch |
| Formatting | git diff --check passes for the port |

The isolated operator executable compiles the exact before/after kernel bodies with the repository CUDA helpers. Shapes cover8/32/128/256/288/384/512/576 experts,1/6/16/33/512 rows, softmax/sigmoid/sqrt-softplus/delayed-softmax, bias and normalization variants. Fixtures cover finite values, ties, one NaN bias, lane-strided NaN biases, all-NaN biases, mixed infinities/NaN, and delayed logits +Infinity with bias -Infinity. The CPU oracle checks exact selected IDs and finite output weights within2e-5 +2e-4*abs(reference); normal-input old/new comparison is bitwise. Output guards and unused ID padding are checked. All-NaN logits or every possible nonfinite normalization case are outside these fixtures; the patch is not a blanket promise to make all invalid routing inputs finite.

Devices were selected explicitly by UUID with CUDA_DEVICE_ORDER=PCI_BUS_ID: GPU-9e5fb796-93b1-b72d-1fd8-438d93aac573 and GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783. No obsolete GPU permutation was used. No compute application was present before the checks. The initial restricted NVIDIA query failed; the elevated read-only query succeeded before any test ran.

## Evidence and remaining queue

Local archive: /home/sergei/_my_old/_my_deepseek41/upstream-31005-20261110. It contains the upstream patch, original/candidate CUDA source, production compile argv/log/exit, operator generator/source/executable/build log, per-device outputs, source hashes, preservation check and SHA256 manifest. Only this REPORT.md is intended for Git; raw evidence remains local.

These are compiler and kernel-level gates, not full ggml graph-fusion, server, DeepSeek/GLM model, state, PPL or PP/TG acceptance. Installed libraries and presets are unchanged. The user subsequently requested committing this isolated port and report; unrelated working-tree changes are excluded. Next in the existing upstream queue is the paired checkpoint/rollback adaptation #31350 + #31368, followed by #31584 and the remaining individually scoped changes. Runtime tests for later stages require their own current baseline and device admission.
