# Qwen3.8-Flash-Next: Local Fix Plan

Start date: 2026-09-10
Status: the fix plan is complete; stage 10 and the additional recurrent rollback bug #28019 have been verified
Source branch: `qwen4exp/mtp`
Source commit: `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89`
Current working repository: `/home/serxxx/Github/llama.cpp`
Current working branch: `my_build_qwen4next`
Plan and original-port repository: `/home/serxxx/_my_qwen4next/llama.cpp`, branch `work/qwen38-fixes`

## Current Working Configuration

### Q8 target + self-contained Q8 MTP with autofit

The successful configuration used on 2026-09-11 is:

- binary: `/home/serxxx/_llama_cpp/glm53_qwen38/llama-server`;
- target: `/home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf`;
- self-contained draft: `/home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`;
- model storage: `/mdd/D2W` on NVMe `/dev/nvme0n1p1` (MSI M560 2TB);
- total context `524288`, `parallel=2`;
- `batch=2048`, `ubatch=512`;
- `fit=on`, `fit-target=3072`;
- `load-mode=none`, `lazy-mode=auto`, `n-cpu-moe=0`;
- `spec-draft-n-max=3`.

Autofit without an explicit KV quantization completes the full startup in `64.39 s`. A long prompt
from the unchanged `src/llama-model.cpp` produced `42831 tokens / 689.49 tok/s` on the first cold pass
and `42836 tokens / 837.58 tok/s` on the second warm pass. The difference is consistent with warming
the lazy Engram/page cache. Two concurrent 1024-token decodes completed at `30.19/30.51 tok/s`, and
a separate 4096-token decode completed at `60.02 tok/s`; there was no late `cuMemCreate`, CUB, or
other runtime OOM.

With `CUDA_VISIBLE_DEVICES=2,1,0,5,4,3`, post-decode VRAM on physical GPUs 0-5 was
`377,45293,45475,16161,45589,19877 MiB`, leaving `48132,3216,3034,3893,2920,4248 MiB` free.
Physical GPU0, which corresponds to logical CUDA2, holds only the CUDA context and receives no
weights. The prefill sampler showed stable usage without an additional temporary peak.

Before the fix, explicitly setting `-ctk q8_0 -ctv q8_0` reproducibly triggered an infinite MoE
autofit loop: attempting to take a partial layer from a partition with `n_part=0` caused unsigned
underflow. The fitter now moves only real partial layers and checks the invariants. The second bug was
cached self-contained MTP estimation: the candidate tensor split changed, but the draft continued to
be accounted for on its original GPU. The self-contained draft is now remeasured using the same
candidate split that it later inherits during the real load.

The verified candidate from `/home/serxxx/Github/llama.cpp/build-glm53/bin/llama-server`, with
`-ctk q8_0 -ctv q8_0` added, completed autofit in `16.75-16.93 s`, selected split
`15,14,14,4,1,1`, and fully loaded in `49.33-49.57 s` without startup OOM. MTP was estimated and
actually placed on logical CUDA4: `2647 MiB` of weights, `1152 MiB` of KV, and `4049 MiB` of compute.
The fix is recorded in target commit `c5f079ab3`, but it has not yet been installed into
`/home/serxxx/_llama_cpp/glm53_qwen38/llama-server`.

Q8 KV runtime validation: a short request reached `55.86 tok/s`, with acceptance
`58/82 = 0.7073`; the `42882`-token prompt from `src/llama-model.cpp` reached prefill
`844.23 tok/s`, decode `40.88 tok/s`, and acceptance `62/96 = 0.6458`. Two concurrent requests with
256 output tokens reached `33.35/32.61 tok/s`, with acceptance `153/213 = 0.7183` and
`146/206 = 0.7087`. Health remained `ok`, with no startup or runtime OOM. Post-load VRAM on physical
GPUs 0-5 was `1731,45013,44661,13763,44563,14739 MiB`, leaving
`46778,3496,3848,6291,3946,9386 MiB` free.

```bash
/home/serxxx/_llama_cpp/glm53_qwen38/llama-server \
    -m /home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
    --model-draft /home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    -fit on \
    --fit-target 3072 \
    -c 524288 \
    -np 2 \
    -b 2048 \
    -ub 512 \
    -fa on \
    --load-mode none \
    --lazy-mode auto \
    --n-cpu-moe 0 \
    -t 20
```

### Q4 target + shared Q8 MTP with manual placement

The verified text configuration uses a Q4 target and shared Q8 MTP:

- target: `/home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`;
- draft: `/mdd/wfree/models/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`;
- total context: `524288`;
- `parallel=2`, giving `262144` tokens per slot;
- manual tensor split: `27,24,27,24,8,6`;
- `batch=512`, with `ubatch=128` required for this configuration;
- autofit disabled: the verified manual placement is used.

```bash
/home/serxxx/Github/llama.cpp/build-glm53/bin/llama-server \
    -m /home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --model-draft /mdd/wfree/models/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-draft-n-max 5 \
    -ngl all \
    -ngld all \
    -ts 27,24,27,24,8,6 \
    -fit off \
    -c 524288 \
    -np 2 \
    -b 512 \
    -ub 128 \
    -fa on \
    --lazy-mode auto \
    -ctk q8_0 \
    -ctv q8_0 \
    -t 20 \
    --no-warmup \
    --no-ui \
    --host 127.0.0.1 \
    --port 8080
```

The `--host` and `--port` values can be replaced by parameters supplied by an external launcher or
proxy. For vision, also add
`--mmproj /home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf`; multimodal
validation was completed in stage 10 and uses a target-only fallback on media batches.

### Engram/PLE placement

The large Qwen3.8 Engram table is stored in `per_layer_token_embd.weight`. The current configuration
explicitly uses `--lazy-mode auto`: because the table is larger than 4 GiB and is marked
`TENSOR_READ_LAZY`, the loader always assigns it to a CPU buffer and reads the required rows through
mmap. The table consumes no VRAM; accessed pages enter normal system RAM through the page cache, while
the rest remains disk-backed. An `-ot` rule is unnecessary for a lazy tensor and does not affect its
placement. Confirm this line during startup:

```text
tensor per_layer_token_embd.weight (...) lazy read enabled
```

To keep the entire table resident in system RAM instead of reading it on demand from SSD, use the
following only when enough memory is available:

```bash
--load-mode none \
--lazy-mode off \
-ot '^per_layer_token_embd\.weight$=CPU'
```

Expected additional resident RAM is approximately `29 GiB` for the Q4 table and `51 GiB` for Q8,
plus other CPU weights and service buffers. This fully resident configuration has not been validated;
the primary configuration is `--lazy-mode auto`, which guarantees that Engram does not use VRAM.

Critical notes for this configuration:

- in the Q4 + shared setup, `-ub 512` with MTP QSA gets a startup OOM while reserving the compute
  graph; `-ub 128` loads and completes concurrent decode. This limitation does not reproduce in the
  Q8 + self-contained autofit configuration above;
- two concurrent requests with 11012 prompt tokens completed without OOM or cross-slot state mixing;
- repeated concurrent runs reached acceptance `0.52941-0.60000` and generation
  `29.43-40.49 tok/s` per slot;
- direct A/B with a 21401-token prompt: dense MTP reached `0.69485` and `63.28 tok/s`, while MTP QSA
  with Top-K sharing reached `0.73828` and `63.69 tok/s`. This is a single run, so the acceptance
  difference is indicative only;
- the QSA threshold-crossing test with a 2041-token prompt passed: acceptance `0.55405`, generation
  `57.17 tok/s`;
- runtime rollback to dense MTP: `QWEN4EXP_MTP_QSA=0`;
- disable reuse of selected MTP blocks with: `QWEN4EXP_MTP_TOPK_SHARE=0`;
- moving the shared draft to a separate `--device-draft CUDA3` is not supported because the borrowed
  `output.weight` remains on target device CUDA5, which is absent from the draft scheduler;
- Engram/PLE loads lazily into a CPU/disk-backed buffer through `--lazy-mode auto` and consumes no VRAM.

### Rollback and diagnostic switches

Set all environment variables before starting the process: some values are cached when the model or
CUDA graph is built for the first time.

| Disabled component | Switch | Result |
|---|---|---|
| All speculative/MTP execution | omit `--model-draft` and `--spec-type draft-mtp` | clean target-only baseline; `--spec-type none` alone is insufficient when a draft is provided because its type can be inferred from GGUF |
| QSA inside the Qwen4Exp MTP head | `QWEN4EXP_MTP_QSA=0` | dense MTP attention instead of indexed attention; primary target QSA remains enabled |
| Reuse of selected MTP QSA blocks | `QWEN4EXP_MTP_TOPK_SHARE=0` | QSA remains enabled, but block selection is recomputed on every draft step |
| Gather-based QSA decode | `QWEN4EXP_QSA_GATHER=0` | previous masked QSA path without compact K/V gather |
| Adaptive MTP draft length | `LLAMA_MTP_ADAPTIVE=0` | fixed limit from `--spec-draft-n-max` instead of shortening inefficient late draft steps |
| CUDA radix-select TOP_K fallback | `GGML_CUDA_TOPK_RADIX_SELECT=0` | generic CUB/argsort fallback; effective in builds without `cub::DeviceTopK` |
| CUDA fused MoE down reduction | `GGML_CUDA_MOE_DOWN_REDUCE=0` | generic MMVQ path for supported low-bit expert-down operations |
| Autofit | `-fit off` plus verified `-ngl`/`-ngld`/`-ts` | manual placement as an operational fallback if the fitter regresses |

`GGML_CUDA_TOPK_ARGSORT=1`, recorded in an early version of the plan for PR #28671, does not exist in
the current code; its actual replacement is `GGML_CUDA_TOPK_RADIX_SELECT=0`. The switches
`GGML_CUDA_TOPK_TEMPORAL=0`, `GGML_CUDA_LID_PREFILL_TILED=0`, and `LLAMA_MTP_DEVICE_DRAFT=0` belong
to GLM-specific paths and do not change the current Qwen3.8 host-driven MTP graph.

## Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` completed and verified
- `[!]` blocked or a regression was found
- `[?]` requires a decision after measurement
- `[-]` deliberately excluded or superseded by a more complete later validation

## Objective

Produce a reproducible local llama.cpp version for Qwen3.8-Flash-Next with:

- correct model mathematics;
- accelerated CUDA TOP_K and gather-based QSA decode;
- minimal unnecessary VRAM allocations;
- working self-contained MTP;
- target configuration `ctx-size=524288`, `parallel=2`;
- controlled speed, acceptance, and quality after every change.

The primary integration order was set by the user:

1. GDN fix;
2. PR #28671;
3. PR #28213;
4. subsequent fixes found by our reviews.

After fixing the indexer V-cache, all completed changes are transferred to a separate, more stable
working branch owned by the user. That branch already contains MTP and `parallel` fixes validated on
another model. They must be validated again and adapted where necessary for Qwen3.8-Flash-Next.

Each stage is performed separately. Work moves to the next stage only after building, testing, and
recording the results. Commits are created only after separate user approval. No push is performed.

## Working Paths and Models

| Purpose | Path |
|---|---|
| Primary working Git repository | `/home/serxxx/Github/llama.cpp`, branch `my_build_qwen4next` |
| Plan and original-port repository | `/home/serxxx/_my_qwen4next/llama.cpp`, branch `work/qwen38-fixes` |
| Previous build copy, no longer used for changes | `/home/serxxx/_llama_cpp/unsloth_qwen4/llama.cpp-qwen4exp-mtp` |
| Models | `/home/serxxx/.models/unsloth/Qwen3.8-Flash-Next-GGUF` |
| Q4 target | `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` |
| Q8 target | `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` |
| MTP draft | `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` |
| Shared MTP draft for the current configuration | `/mdd/wfree/models/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` |
| MMProj | `mmproj-F16.gguf` |

Current edits, Git operations, builds, and measurements are performed in `/home/serxxx/Github/llama.cpp`.
The `/home/serxxx/_my_qwen4next/llama.cpp` repository stores this plan and the history of the original
ports. The `_llama_cpp/unsloth_qwen4` copy is no longer used for development.

## Recorded Starting Point

From the available user logs:

| Metric | Initial value |
|---|---:|
| Draft acceptance | approximately `0.51` in the latest successful run |
| Earlier observed acceptance | `0.26104`, 1176 / 4505 |
| Decode speed in one run | approximately `46.07 tok/s` |
| Target total context | `524288` |
| Target slot count | `2` |
| Context per slot | `262144` |
| MTP | self-contained Q8, not shared |

These numbers are not a clean benchmark. Comparisons require identical prompts, seed, sampling, GPU
split, and server parameters.

## Stage 0. Reproducible Baseline

- [x] Select a single working Git directory.
- [x] Create a local working branch from `d1a92352` without modifying the source `qwen4exp/mtp` branch.
- [x] Record `git status`, HEAD, and the diff before starting work.
- [x] Record the NVIDIA driver, CUDA toolkit, CUB/CCCL, and GPU compute capability versions.
- [x] Determine which CUDA TOP_K path was actually built: the CUB argsort fallback is used because
  CCCL `3.0.1` is below the `3.2` threshold for `CUB_TOP_K_AVAILABLE`.
- [x] Perform a clean CUDA build so that `llama-server --version` no longer reports `commit unknown`.
- [x] Run `test-llama-archs` on CPU and CUDA.
- [x] Run `test-backend-ops -o TOPK_QSA` on CUDA.
- [x] Run the minimum required existing tests for the affected subsystems.
- [x] Preserve the exact server command: the current Q8 autofit and Q4 manual presets are shown at
  the beginning of this document.
- [x] Preserve an identical prompt set and sampling parameters: the final matrix uses deterministic
  short prompts and a structured 148k prompt containing three markers.
- [x] Measure without and with MTP: speed, acceptance, VRAM after load, peak VRAM, and time to first
  decode; the results are consolidated in stages 9, 10, and the final table.
- [x] Establish Q4 and Q8 baselines at both short and long context: short and structured 148k runs
  were completed for both quantizations.
- [x] Separately reproduce two simultaneous requests with distinguishable prompts at `parallel=2`:
  ALPHA/BETA, a 2x1024 stress test, and concurrent vision+text passed on Q4 and Q8.

Completion criterion: the results reproduce at least twice, the build corresponds to a known commit,
and the baseline is recorded in the table below.

Environment recorded on 2026-09-10:

- NVIDIA driver `580.159.04`;
- CUDA toolkit / nvcc `13.0.88`;
- CUB `300001`, which means CCCL `3.0.1`;
- GPUs: four compute capability `8.9` devices and two `8.6` devices;
- clean CUDA build: `build-cuda`, Release, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`,
  `GGML_CUDA_GRAPHS=ON`, architectures `86-real;89-real`;
- built binary version: `0.3.0-dev`, build `10802`, commit `d1a92352c`.

The historical performance baseline for source `d1a92352` is unsuitable for strict A/B because its
prompts and sampling differ. A reproducible baseline for the current stable branch was obtained later
in stages 9-10 and should be used for future comparisons.

## Stage 1. GDN Normalization Fix

Source: [upstream PR #28068](https://github.com/ggml-org/llama.cpp/pull/28068), merged commit
`5fdfa6282936576d2f352d4b97f397a109f207a6`.

Goal: replace the old normalization

```text
x / max(sqrt(sum(x^2)), eps)
```

with the formula used in training and the reference implementation:

```text
x * rsqrt(sum(x^2) + eps)
```

- [x] Compare the complete upstream diff against the local branch.
- [x] Check for conflicts with qwen4exp/MTP changes.
- [x] Port upstream helper `build_gdn_l2_norm` and its calls without independently changing the formula.
- [x] Decide whether to port the entire commit or only the minimum qwen4exp subset: the minimum
  qwen4exp scope was selected; the seven other models in the upstream commit are unchanged.
- [x] Build CPU and CUDA variants.
- [x] Run architecture and backend tests.
- [x] Run a short Q4 and Q8 smoke test without MTP.
- [x] Run a short Q4 and Q8 smoke test with self-contained Q8 MTP.
- [-] Do not preserve a separate strict performance A/B for GDN alone: the old runs used different
  prompts, while GDN correctness and final performance were verified by later regression and runtime matrices.
- [x] Request user approval before creating the local commit; commit `36f1df783` was created after review.

Completion criterion: qwen4exp contains no remaining call to the old `ggml_l2_norm` for GDN q/k,
tests pass, and generation is coherent.

Stage 1 validation results:

- CPU build `test-llama-archs`: `qwen4exp` NMSE `0.00e+00`, roundtrip `OK`;
- CUDA build `test-llama-archs` on RTX 4090: `qwen4exp` NMSE `8.76e-08`, roundtrip `OK`;
- CUDA `test-backend-ops -o TOPK_QSA`: `4/4` tests `OK`;
- Q4 without MTP: coherent generation, prompt `96.7 tok/s`, generation `61.3 tok/s`;
- Q8 without MTP: answer `the sky is blue.`, prompt `55.2 tok/s`, generation `48.8 tok/s`;
- Q4 + self-contained Q8 MTP: same answer, prompt `53.1 tok/s`, generation `63.5 tok/s`;
- Q8 + self-contained Q8 MTP: same answer, prompt `68.3 tok/s`, generation `69.2 tok/s`.

Common smoke-test parameters: `ctx=4096`, `parallel=1`, `seed=42`, `temp=0`, reasoning off, autofit
off, six GPUs. MTP used `--spec-type draft-mtp --spec-draft-n-max 5 -ngld all`. The answer is too
short for representative acceptance; these speeds cannot be compared directly with the original
server measurements.

## Stage 2. CUDA Radix-Select TOP_K, PR #28671

Source: [upstream PR #28671](https://github.com/ggml-org/llama.cpp/pull/28671), PR commit when the
plan was written: `ff2b436`. The PR was not merged at that time.

Purpose: replace the full segmented radix sort in the CUB fallback with radix-select for wide rows.
For a 131k context, the PR author measured TOP_K dropping from 5.1 to 0.25 ms/token and an overall
decode improvement of approximately 13-18%.

Limitation: the patch is active only in builds without `cub::DeviceTopK`, which means CCCL below 3.2
in the current code. Local CCCL `3.0.1` uses exactly the CUB fallback being fixed.

- [x] Fetch the exact diff of the current PR head and record its SHA:
  `ff2b436762b3c412ed8edfcef401e32629ad4c4c`.
- [x] Check whether the PR changed before porting: on 2026-09-10 it was open and still contained the
  single commit `ff2b436`.
- [x] Confirm the active local TOP_K code path: CCCL `3.0.1`, with `CUB_TOP_K_AVAILABLE` undefined.
- [x] Port the patch on top of the GDN stage with a minimal diff; the final `top-k.cu` blob matches
  the PR (`cf4122b27`).
- [x] Preserve a runtime kill switch; in the final branch it is named `GGML_CUDA_TOPK_RADIX_SELECT=0`.
- [x] Run all existing TOP_K backend tests: `525/525` with radix-select and `525/525` with argsort.
- [x] Test wide rows: the existing set covers 8192, 32768, 131072, 262144, 524288, and 524299 elements.
- [x] Check repeated selection on equal values at least 20 times: the exact block-selection fix in
  stage 4 passed `20/20` on CUDA0 and `6/6` on every GPU.
- [x] Check runtime VRAM and absence of late `cuMemCreate` OOM: stages 9-10 completed long and
  concurrent Q4/Q8 decode without late OOM.
- [x] Run kernel A/B using the former switch name; use `GGML_CUDA_TOPK_RADIX_SELECT=0` for the current binary.
- [x] Measure short, 31k, 62k, 131k, and the maximum achievable context: kernel A/B was supplemented
  by the 32k/64k/128k runtime depth grid, structured 148k, and full `ctx=524288`, `parallel=2` allocation.
- [x] Request user approval before creating the local commit; commit `bc495e524` was created after review.

Warning: radix-select, like `cub::DeviceTopK`, may select nondeterministic elements when cell scores
are equal. Qwen4exp originally expanded one block score into four equal cell scores. Therefore, the
stage 2 result could not be used as a final production build before stage 4. In the current branch,
the fallback is enabled with `GGML_CUDA_TOPK_RADIX_SELECT=0`.

Completion criterion: backend tests pass, acceleration is measured on local hardware, the kill
switch works, and no new OOM occurs.

Stage 2 results on RTX 4090, `k=16`, one row:

| Width | Radix-select | Argsort | Speedup |
|---:|---:|---:|---:|
| 4096 | 20.45 us | 20.38 us | 1.00x; both runs use argsort below the threshold |
| 8192 | 32.84 us | 84.56 us | 2.57x |
| 32768 | 32.30 us | 88.29 us | 2.73x |
| 65536 | 36.26 us | 89.68 us | 2.47x |
| 131072 | 41.15 us | 92.33 us | 2.24x |

Additional regressions: CUDA `test-llama-archs` for `qwen4exp` produced NMSE `8.49e-08`, roundtrip
`OK`; CUDA `TOPK_QSA` was `4/4 OK`. The initial kernel-level validation was later supplemented by
the end-to-end matrices in stages 7, 9, and 10.

## Stage 3. Gather-Based QSA Decode, PR #28213

Source: [upstream PR #28213](https://github.com/ggml-org/llama.cpp/pull/28213). The PR was not merged
at that time.

Purpose: during single-token decode, gather only the selected QSA K/V into a compact buffer instead
of running dense attention over the entire context with a mask. Published measurements showed gains
of approximately 6% at 31k, 19% at 62k, and 50% at 130k.

- [x] Fetch the exact PR head SHA immediately before porting:
  `beed2f78ac42cf16710b763e6f3ba20665c6d233`.
- [x] Review discussion and unresolved comments: the PR was open with one commit and had not yet
  received mandatory review; the earlier `build_attn_mha` incompatibility had been fixed by the author's rebase.
- [x] Compare the PR diff with local qwen4exp MTP code; signatures are compatible and local GDN/MTP
  changes remain intact.
- [x] Port the changes after #28671 while preserving MTP and local QSA changes.
- [x] Preserve kill switch `QWEN4EXP_QSA_GATHER=0`.
- [x] Confirm that prefill and batched decode remain on the expected path: gather requires
  `n_tokens == n_stream` and `n_kv >= 4*width`; stable A/B prefill reached 1187.3 and 1189.4 tok/s.
- [x] Run CPU and CUDA tests: both builds pass; `test-llama-archs` for qwen4exp gives CPU roundtrip
  `OK`, CUDA NMSE `8.91e-08`, roundtrip `OK`.
- [x] Run correctness A/B with forced stable TOP_K; the current fallback name is
  `GGML_CUDA_TOPK_RADIX_SELECT=0`.
- [-] Do not add a separate runtime logits dump between masked and gather: stage 4 builds both paths
  from the same verified active set, the masked/gather regression passes, and the runtime rollback
  switch plus deterministic greedy smoke are retained.
- [x] Check the gather threshold and absence of short-context regressions: below `4*width`, the
  previous masked graph remains active; the short qwen4exp architecture test passes.
- [x] Measure 4k, 31k, 62k, 131k, and long context: short, 32k/64k/128k depth grid, structured 148k,
  and `ctx=524288`, `parallel=2` runtime are complete; an actual 262k prompt was deliberately excluded
  later in stage 7.
- [x] Check MTP acceptance and total speed, not only target decode speed: stages 7 and 10 contain
  direct MTP QSA A/B, a depth grid, and final Q4/Q8 acceptance/speed.
- [x] Test Q4 and Q8: Q4 A/B and Q4+MTP passed; Q8 gather smoke passed with partial CPU offload.
- [x] Request user approval before creating the local commit; `6593b0abb` was created after review.

Completion criterion: gather can be disabled at runtime, results match the old path with a fixed set
of QSA indices, and long-context speed improves without reducing acceptance.

Stage 3 validation results:

- Q4, `ctx=32768`, `parallel=1`, Q8 KV, `temp=0`, without MTP, fixed prompt, stable argsort: masked
  `53.5 tok/s`, gather `55.6 tok/s`, or `+3.9%`;
- in the 128-token pair, output files matched through byte 50220; after `Assistant:`, the common answer
  prefix was 575 bytes, after which accumulated numerical differences changed the open-ended greedy continuation;
- an additional 64-token pair with fast radix TOP_K: masked `53.2 tok/s`, gather `58.0 tok/s`, or
  `+9.0%`; the short beginning of the answer matched;
- Q8 gather smoke: `ctx=32768`, 32 GPU layers and 16 CPU layers, generation `8.5 tok/s`, coherent
  answer, no runtime OOM; this is not a benchmark;
- full manual Q8 GPU offload did not fit with the tested split, and row split is unsupported by this
  CUDA build; this is a placement limitation of the 188-GB Q8 model, not a gather error;
- Q4 + self-contained Q8 MTP: long smoke passed at generation `64.8 tok/s`; because target
  verification is multi-token, gather in the current PR does not accelerate MTP.

The context depths and representative MTP measurements initially missing here were completed in
stages 7 and 10. A separate debug logits dump was not added: block-selection correctness is tested
directly, and the old masked path remains available as a runtime rollback.

## Stage 4. Correct QSA Block Selection

Related problems:

- local code always selected `indexer_top_k + r - 1 = 2051` cells instead of `2048 + tail`;
- [issue #28497](https://github.com/ggml-org/llama.cpp/issues/28497): nondeterministic CUDA TOP_K
  selection for equal cell scores;
- #28671 likewise did not guarantee stable ordering of equal values.

Preferred design: perform TOP_K over block scores, expand the selected complete blocks into cell
indices, and append only the real incomplete tail. This is closest to the Transformers reference and
prevents cutting a block at the TOP_K boundary.

- [x] Agree on the exact block-selection design with the user before implementation: `512` block
  scores, expansion by `r=4`, and a separate tail without epsilon.
- [x] Evaluate variable-tail representation in a static ggml graph: physically allocate `r` slots;
  activate `0..r-1`, mask the rest, and pad gather to the FA width.
- [x] Do not alter scores with an arbitrary epsilon: no epsilon is used.
- [x] Implement TOP_K over complete blocks with `ggml_argsort_top_k`, then expand block IDs into cell
  IDs and explicitly append the tail.
- [x] Test tails `0`, `1`, `2`, and `3`.
- [x] Test contexts `2047`, `2048`, `2049`, `2050`, `2051`, and long context `32771`.
- [x] Verify that no complete block is partially selected: the new `QSA_BLOCK_SELECT` test validates
  every active group of four cells.
- [x] Repeat the CUDA test at least 20 times: `20/20` on CUDA0; full `6/6` on each of six GPUs.
- [-] Exclude direct logits comparison with the Transformers reference: the selection structure and
  tail match the reference code, but the original HF checkpoint is unavailable locally; block-level
  and real-GGUF regressions pass.
- [x] Repeat #28671 and #28213 regressions after changing the TOP_K shape: the old CUDA `TOPK_QSA`
  is `4/4` on every GPU; the new masked/gather test is `6/6`; real Q4 A/B produced the same first 16
  greedy tokens.
- [x] Request user approval before creating the local commit; `6872b63c1` was created after review.

Completion criterion: the selected set matches the reference, a complete block is never split, and
repeated greedy runs are deterministic within backend guarantees.

Stage 4 validation results:

- block selection operates on `2048/4 = 512` complete blocks; selected block IDs are expanded through
  the cache-cell map;
- the tail is stored separately in four physical slots, but only `0..3` are active; inactive tail
  entries and FA padding receive a negative mask;
- masked and gather paths are built from the same block selection and produce the same active set in the test;
- CUDA `QSA_BLOCK_SELECT`: contexts `2047`, `2048`, `2049`, `2050`, `2051`, `32771` are `6/6` on
  each of six GPUs; the repeated CUDA0 cycle is `20/20`;
- CPU/CUDA `test-llama-archs -a qwen4exp` passed, with CUDA NMSE `8.54e-08`, roundtrip `OK`;
- real Q4, long report prompt, `ctx=32768`, Q8 KV, `temp=0`: gather and masked matched for a 16-token
  greedy smoke; both reached `54.6 tok/s` (not a benchmark);
- review removed the obsolete tail-score boost `+1e9`: once the tail is appended explicitly it is
  unnecessary and could compensate for the finite mask of a synthetic block.

## Stage 5. Remove the Unused Indexer V-Cache

Source: [upstream PR #28330](https://github.com/ggml-org/llama.cpp/pull/28330), head commit
`b12a411b43aa1e2f7f5856c876817f7ae2d2a770`, merged on 2026-09-10 as
`311d4211bf1611ff7ca6b67035a4a07c79766efc`.

The qwen4exp indexer uses only K, but the generic `llama_kv_cache` also allocated V with width 256.
For F16 this wasted 6144 bytes/token across 12 QSA layers. At 262144 tokens per slot, that is about
1.5 GiB per stream and about 3 GiB for two non-unified streams.

- [x] Fetch the current PR diff and inspect changes after `b12a411`: the head was unchanged and the
  PR had been merged.
- [x] Determine whether the patch is restricted to qwen4exp or changes the general KV API: the diff
  consists of four lines in `llama-memory-hybrid-idx.cpp` only.
- [x] Port the patch and inspect every place that may assume non-null V: the upstream diff was ported;
  the generic K-only path, state I/O, and memory accounting were checked in code and at runtime.
- [x] Run save/load-state and hybrid/indexer-memory tests.
- [x] Compare the memory breakdown before and after.
- [x] Repeat `ctx-size=524288`, `parallel=2` startup.
- [x] Repeat autofit and record its duration and selected GPU split.
- [x] Test Q4/Q8, MTP on/off, and multimodal startup.
- [x] Request user approval before creating the local commit: source commit `d1946ac89` was created
  after review; the stable-branch port is `9f698750f`.

Stage 5 validation results:

- the exact four-line upstream diff sets both MLA widths in the indexer's hparams copy; the existing
  K-only `llama_kv_cache` path then creates no V;
- on Q4 with F16 KV, `ctx=32768`, `parallel=1`, the indexer shrank from `288 MiB`
  (`96 MiB K + 192 MiB V`) to `96 MiB` (`96 MiB K + 0 MiB V`);
- at `ctx-size=524288`, `parallel=2`, two streams of `262144` cells are created: the F16 indexer uses
  `1536 MiB` instead of the expected old `4608 MiB`, saving `3072 MiB`;
- with Q8 KV in the same target configuration, the indexer uses `816 MiB` and contains no V, saving
  `1632 MiB` compared with the former layout;
- CPU/CUDA `test-llama-archs -a qwen4exp` passed, CUDA NMSE `8.74e-08`, roundtrip `OK`; CUDA
  `QSA_BLOCK_SELECT` was `6/6`, `TOPK_QSA` was `4/4`;
- real Q4 passed the complete `test-save-load-state`: `8/8`, including host/device copy and scatter restore;
- Q4 starts successfully at `ctx-size=524288`, `parallel=2`, both without MTP and with self-contained
  Q8 MTP plus `mmproj-F16`; multimodal text/vision/video modes are registered;
- the old manual split `24,24,24,10,24,12` is unsuitable for long context because the compute buffer
  does not fit on 20-GB CUDA4; split `24,24,24,24,8,12` starts Q4 successfully;
- Q4 autofit completed its calculation in `0.38 s` without changing parameters; total warm startup
  took about `24 s`;
- Q8 with Q8 KV starts successfully at `ctx-size=524288`, `parallel=2`, both without MTP and with
  self-contained Q8 MTP plus `mmproj-F16`; autofit for the full combination completed in `0.79 s`
  without changing parameters;
- the first Q8 startup took `4:11` because it actually read and placed the 188-GB model; a repeated
  warm Q8 + MTP + mmproj startup loaded in about `41 s`, so the delay was not an autofit loop;
- short Q8 + MTP smoke: `47/79` draft tokens accepted, acceptance `0.59494`, mean length `3.94`,
  generation `69.44 tok/s`; the sample is too short for strict quality or speed comparison.

Completion criterion: the indexer V-cache is not allocated, memory use falls by the expected amount,
and save/load plus QSA work without regressions.

## Stage 5.5. Transfer to the Stable Working Branch

After completing GDN, #28671, #28213, QSA block selection, and the V-cache fix, transfer the entire
verified change set to another user-owned working branch based on a more stable llama.cpp version.

The target branch already contains MTP and `parallel` fixes, but they were tested on a different
model. Their presence is not evidence of qwen4exp correctness: Qwen3.8-Flash-Next additionally uses
hybrid recurrent state, QSA, PLE, and a separate MTP graph.

- [x] Record the target repository path, branch name, remotes, and exact base SHA.
- [x] Check `git status` in both working trees and preserve their initial diffs.
- [x] List the target branch's MTP/parallel fixes relative to its upstream base: key commits
  `874703118`, `803b76c92`, and `ccdf0ae2c` were reviewed in stage 6.
- [x] Separate generic MTP/parallel changes from changes specific to the previously tested model: the
  generic backend-state infrastructure was reused, while the GLM5Next-only device loop was not ported to Qwen.
- [x] Check intersections with `common/speculative.cpp`, server slot logic, recurrent memory, graph
  inputs, and KV cache: covered by stages 6, 8.3, 8.4, and the final parallel matrix.
- [x] Record the order of individual ports; do not transfer everything as one opaque diff.
- [x] Port one verified change at a time: GDN was already in the base; #28671 `36279d50d`; #28213
  `58670995e`; QSA correctness `96990f5a7`; no indexer V-cache `9f698750f`.
- [x] Build and run the minimum tests for each stage after every port.
- [x] Do not transfer build artifacts, old CMake caches, or parameters specific to the previous hardware.
- [-] Do not reconstruct a separate historical baseline of the stable branch before its MTP/parallel
  fixes: it was not recorded before integration; future comparisons use target-only controls and
  runtime rollback switches on one current base.
- [-] Do not perform a monolithic A/B of the entire stable branch with and without the old
  MTP/parallel fixes: the fixes were tested by component, synthetic regressions, and the final
  target-only/MTP matrix.
- [x] Test Q4 and Q8 targets with self-contained Q8 MTP: Q4 is recorded in `8ac6982b1`; Q8 later
  passed short, structured 148k, and `parallel=2` tests with corrected autofit.
- [x] Test borrowed/shared Q8 MTP: Q4 accepted the same `41/56` draft tokens as self-contained;
  `98a58b551` was created after review.
- [x] Test `parallel=1` before moving to simultaneous requests.
- [x] Run a two-slot isolation test at `parallel=2` with sharply different prompts: ALPHA/BETA and
  ORBIT/RECIPE do not mix.
- [x] Check acceptance, speed, VRAM, rollback, prompt cache, and multimodal operation: final results
  are recorded in stages 6, 9, and 10.
- [x] If fixes depend on the previous model's architecture, adapt them only after determining the
  cause and agreeing on the design: Qwen uses host-driven MTP, a separate QSA path, and an
  architecture-gated multimodal fallback.
- [x] Request user approval before every local commit.

Completion criterion: all five preceding fixes exist in the stable branch as traceable individual
changes, the baseline is repeated, and existing MTP/parallel fixes are validated specifically on
Qwen3.8-Flash-Next.

Recorded target base:

- repo: `/home/serxxx/Github/llama.cpp`;
- branch: `my_build_qwen4next`;
- remotes: `origin = git@github.com:smalinin/llama.cpp.git`,
  `upstream = https://github.com/ggml-org/llama.cpp.git`;
- initial branch SHA: `271235bb630a7f258a47c18a379aa2ef4d8d1557`;
- Qwen self-contained MTP port: `8ac6982b1`, Q4 `ctx=4096`, `parallel=1`, acceptance `0.73214`
  (`41/56`), mean length `3.05`, generation `72.18-78.23 tok/s`;
- Qwen shared MTP borrowing: `98a58b551`; embeddings and output weights are borrowed from the
  target, while KV memory remains separate; deterministic A/B matched self-contained (`41/56`,
  acceptance `0.73214`);
- shared MTP with `--fit on`: load completed in about 23 s; the expected warning during trial draft
  memory measurement did not prevent startup; smoke produced `40/53`, acceptance `0.75472`, and
  generation `93.36 tok/s`;
- before the fix, `test-save-load-state` test 5 failed identically with and without the V-cache fix;
  block-aware device copy is recorded in `2ecc7b194`.

## Stage 6. MTP and `parallel=2`: State Isolation

Problem source: [issue #28286](https://github.com/ggml-org/llama.cpp/issues/28286). An additional
related problem is open [issue #28019](https://github.com/ggml-org/llama.cpp/issues/28019), concerning
restoration of Qwen4Exp recurrent snapshot planes.

Until this stage was completed, it was a production blocker for the target configuration and only
the following modes were allowed:

- MTP with `parallel=1`;
- `parallel=2` without MTP;
- `parallel=2` with MTP only when an external scheduler guaranteed a single active request.

- [x] First fix baseline regression `test-save-load-state` test 5: block-aware device copy is in
  `2ecc7b194`; the Qwen4Exp Q8 KV regression and the real Q4 model pass tests 1-8.
- [x] Fix the GLM5Next reserve failure found by the full matrix: logical sequences and unified KV
  streams are separated in the synthetic memory context; CPU/CUDA GLM5Next, full save/load matrices,
  and architecture tests pass; commit `8a8c83913` was created.
- [x] Then review the stable branch's existing MTP/parallel fixes from stage 5.5: `874703118`,
  `803b76c92`, and `ccdf0ae2c` are present in the target branch; they split backend hidden state by
  sequence/chain/stage, preserve pending MTP state across context shift, and validate
  target/draft/spec positions during prompt reuse.
- [x] Create a reproducible manual test with two simultaneously active slots and sharply different
  prompts: fixed slots 0/1, `cache_prompt=false`, mutually exclusive ALPHA/BETA and ORBIT/RECIPE markers.
- [x] Test target without MTP as a control: concurrent batching also changes greedy output. This is
  documented backend nondeterminism for different batch sizes; no semantic state crosses between slots.
- [x] Test MTP at `parallel=1` as a control: self-contained and shared variants previously produced
  the same `41/56` and acceptance `0.73214`.
- [x] Test with CUDA graphs on and off: two simultaneously occupied slots pass in both modes, with
  acceptance in the expected range.
- [-] Do not add a separate runtime trace for every rollback plane: slot IDs and acceptance/draft
  counts are recorded, and the existing `test-recurrent-state-rollback` now checks plane state directly.
- [x] Inspect shared buffers in the speculative accept/verify loop: ported commit `874703118` stores
  backend hidden output separately by sequence/chain/stage; manual concurrent tests found no overwrite.
- [x] Check the connection to recurrent rollback issue #28019: after fixing an incorrect full/partial
  logits reference, the real Qwen3.8 Q4 model reproduced a genuine `multi-seq split replay` mismatch
  at the first replay position.
- [x] Establish the root cause before applying a fix: for a batch shorter than the rollback window,
  fused GDN wrote only post-token planes `0..n_seq_tokens-1`; plane `n_seq_tokens`, representing the
  input state before the batch, remained from an earlier decode.
- [-] Do not use the initially agreed post-read replication: A/B showed that it did not change the
  failure and added an unnecessary recurrent-cache copy on every checkpoint restore.
- [x] Strengthen existing `test-recurrent-state-rollback`: separate full and partial logits
  references, restore the quantized `seq_cp` reference from an exact checkpoint, and directly compare
  logical R/P/S snapshot rows for Qwen4Exp.
- [-] Do not run a multi-hour concurrent soak as part of the fix work: five consecutive concurrent
  rounds, prompt-cache reuse, 2x1024 stress, and the long-context matrix passed; further soak testing
  is an operational concern.
- [x] Check for semantic transfer between requests: all short and 524k runs preserved the mutually
  exclusive markers of their slots.
- [x] Request user approval before creating the local commit; `26abcbf7e` was created after review.

Completion criterion: independent parallel requests do not affect each other, the test reproduces
the error before the fix, and passes afterward.

### Reproducer #28019, 2026-09-11

The issue remains open. The branch contains merged fix #28123 (`0eadefebd`), which added separate
rollback snapshots for GDN QKV-conv and PLE-conv. The additional test initially reported
`dirty-ctx logits mismatch` incorrectly: the shared reference array had been overwritten by
`PARTIAL_ONLY` results, after which full restore was compared against the partial reference. After
separating the references, dirty-context restore passed without production-code changes.

The corrected test exposed the real #28019 defect: multi-sequence split replay diverged at the first
replay position. Direct comparison of logical recurrent rows localized the difference to
`cache_s_l0`. Fused GDN returns only the last `min(n_seq_tokens, K)` post-token snapshots. For a
three-token batch, planes 0-2 were updated, while plane 3, which must represent the input state before
those three tokens, remained from an earlier batch.

Fix `26abcbf7e` saves the input GDN state in plane `n_seq_tokens` when the batch is shorter than the
snapshot window. The state-file format and in-memory checkpoint format are unchanged. The previously
proposed post-read replication was tested A/B, did not affect the mismatch, and was removed before commit.

Post-fix validation:

- real Qwen3.8 Q4, `-c 512 -b 128 -ub 128`, full GPU offload, split `27,24,27,24,8,6`: checkpoint
  restore, multi-sequence replay, sequence isolation, wildcard removal, and `seq_cp` all passed with
  `max diff 0`;
- synthetic Qwen4Exp on CPU passed the complete test;
- the registered CTest matrix for Qwen35, Nemotron-H, and DeepSeek4 passed `4/4`;
- the strengthened Qwen4Exp snapshot assertion reproduced the `cache_s_l0` difference before the
  production fix and passes afterward.

## Stage 7. MTP QSA and Long-Context Acceptance

The MTP checkpoint contains its own QSA indexer weights, but published Unsloth GGUF files have
`compress_ratio=0` for added layer 48, so the previous graph used dense attention. The loader now
restores ratio 4 from the homogeneous nonzero ratios of the main model, but only when the MTP layer
actually contains `indexer.q_proj.weight`.

- [x] Confirm the presence and shapes of `mtp.layers.0.self_attn.indexer.*` in the source checkpoint
  and GGUF: `k_norm (128)`, `k_proj (2560,128)`, `q_norm (128)`, `q_proj (2560,512)`.
- [x] Check which indexer tensors are actually stored in self-contained and shared MTP GGUF: both
  variants contain all four tensors; shared differs through borrowing embeddings/output, not its QSA indexer.
- [-] Do not run an actual 262k-deep prompt per slot: acceptance was measured at
  2k/32k/64k/128k and structured 148k, while full `524288/parallel=2` allocation and concurrent
  runtime were tested separately.
- [x] Separate the influence of GDN, target QSA, and MTP attention: GDN has its own regression;
  MTP QSA and Top-K reuse can be disabled independently with `QWEN4EXP_MTP_QSA=0` and
  `QWEN4EXP_MTP_TOPK_SHARE=0`.
- [x] Design MTP QSA without duplicating another large cache: use the existing K-only indexer cache
  and retain only 512 I32 block IDs per stream between MTP iterations, about 4 KiB for `parallel=2`.
- [x] Compare dense MTP and QSA MTP for acceptance, speed, and VRAM: direct 21.4k A/B, the
  32k/64k/128k depth grid, and the final runtime matrix are complete; do not interpret the acceptance
  difference from a single run as a quality benchmark.
- [x] Preserve a runtime fallback to dense MTP through validation: `QWEN4EXP_MTP_QSA=0`; a separate
  no-share fallback is `QWEN4EXP_MTP_TOPK_SHARE=0`.
- [x] Request user approval before creating the local commit; `65157dfd4` was created after review.

The implementation uses the target model's common QSA attention path. MTP step 0 computes and saves
block IDs; subsequent speculative steps reuse the same selection and do not repeat indexer
projection/scoring. The cache is stream-local and resets when memory state changes or is restored.

Direct A/B on an identical 21401-token prompt, `ctx=32768`, `parallel=1`, Q4 target + shared Q8 MTP,
320 output tokens:

| MTP attention | Acceptance | Generation | Prompt eval |
|---|---:|---:|---:|
| dense fallback | 0.69485 (189/272) | 63.28 tok/s | 1157.18 tok/s |
| QSA without Top-K sharing | 0.71146 (180/253) | 62.09 tok/s | 1171.07 tok/s |
| QSA with Top-K sharing | 0.73828 (189/256) | 63.69 tok/s | 1171.06 tok/s |

This is one run, so the acceptance difference is indicative only. After the final refactor, a
repeated 11k smoke run produced acceptance 0.64815 (35/54), generation 56.51 tok/s, and confirmed
state reset. A separate 2041-token prompt crossed the 512-block threshold and, after fixing pruned
host input, completed with acceptance 0.55405 (41/74) and generation 57.17 tok/s.

QSA depth grid, `parallel=1`, Q4 target + shared Q8 MTP, `ubatch=128`, synthetic repeated-token prompts:

| Prompt tokens | Acceptance | Generation | Prompt eval | Result |
|---:|---:|---:|---:|---|
| 32708 | 0.63492 (40/63) | 58.92 tok/s | 887.62 tok/s | marker correct |
| 65468 | 0.54955 (61/111) | 39.03 tok/s | 785.50 tok/s | 128 tokens, marker instruction unstable |
| 130907 | 0.58120 (68/117) | 31.92 tok/s | 614.47 tok/s | first decode returned EOS; retry with `ignore_eos=true` returned the marker and 128 tokens |

The depth grid confirms the absence of runtime OOM and working MTP QSA reuse through 128k. These
synthetic prompts are suitable for checking depth and speed, but are not a quality benchmark;
quality evaluation requires structured long-context prompts.

The required `ctx=524288`, `parallel=2` configuration loads with split `27,24,27,24,8,6` and
`ubatch=128`; each slot receives 262144 tokens. Two repeated simultaneous runs with 11012 prompt
tokens completed without OOM or state mixing: acceptance was 0.52941-0.60000, generation
29.43-40.49 tok/s. With `ubatch=512`, MTP QSA graph reservation still gets a startup OOM. Attempting
to move the shared draft with `--device-draft CUDA3` exposed a separate limitation: borrowed
`output.weight` stays on target device CUDA5, which is absent from the draft scheduler.

Completion criterion: select the option with the best end-to-end throughput without worse quality,
based on measurements rather than structural similarity alone.

## Stage 8. Other Correctness and Test Fixes

### 8.1 `llama_memory_recurrent::seq_rm(seq_id < 0)`

- [x] Fix premature rejection of negative `seq_id`: apply the upper-bound check only to nonnegative IDs.
- [x] Verify the `seq_id < 0: match any sequence` contract: the base KV cache now accepts any
  negative value, not only `-1`.
- [x] Add cases to the existing recurrent rollback test: `-1`, `-2`, two sequences, full removal,
  an empty range, and cache reuse.
- [x] Test partial rollback and repeated `seq_rm`: partial wildcard returns `false` without mutation;
  full removal resets stale recurrent tail pointers; repeated full removal succeeds.
- [x] Test the DSV4 wrapper: an empty or reversed range is now a successful no-op before the generic
  prohibition on partial rollback.
- [x] Request user approval before creating the local commit.

Before the fix, the regression test first reproduced `invalid seq_id (-1)`. After removing the early
rejection it found a base-KV assertion on `-2` and stale recurrent tail pointers on the next decode.
After the complete fix, registered Qwen35, Nemotron-H, and DeepSeek4/DSV4 recurrent rollback CTests
pass `4/4`; manual CPU/CUDA variants also pass. `test-save-load-state` passes `2/2` together with the
model-generation fixture.

Priority is medium for the qwen4exp server: its standard path uses nonnegative IDs, but the generic
API contract had been violated.

### 8.2 EOS List in the Converter

- [x] Match `_eos_token_id()` to reference semantics: use the first list item rather than the last.
- [x] Add a small converter test for scalar EOS and an EOS list.
- [x] Request user approval before creating the local commit; target commit `6a413fbcc` was created
  after review.

Behavior is unchanged for the current checkpoint with one EOS. The new unit test calls the real
`Qwen4ExpTextModel._eos_token_id()` and checks scalar `248044` and list `[248044, 248046]`; both cases
pass, and the list returns primary EOS `248044`.

### 8.3 MTP Test Coverage

- [x] Add a NEXTN/MTP configuration to the existing architecture-test infrastructure.
- [x] Ensure that `graph_mtp` is built and executed by existing `test-llama-archs` on CPU and CUDA.
- [x] Add smoke coverage for self-contained and borrowed/shared variants without a new heavyweight test target.
- [x] Test reject/rollback of two consecutive draft tokens, repeat decode, and compare logits.
- [x] Test host-driven `parallel=2`: both sequences match the single-threaded CPU reference.
- [x] Test Qwen QSA MTP: the first step builds a selection, while the second consumes saved block IDs
  without repeating the score calculation.
- [x] Request user approval before creating the local commit; target commit `9b374c470` was created
  after review.

The synthetic Qwen4Exp MTP fixture uses two trunk layers and one NextN layer. The self-contained
variant is tested directly. The shared variant is saved into a temporary GGUF without its own
embeddings or output head, then loaded with `ctx_other`, so it actually exercises the borrowing path.
Rollback removes both speculative positions; repeated decode must have NMSE no greater than `1e-12`.

After cleaning the diff, `test-llama-archs` built and passed. `qwen4exp` passed on CPU and six CUDA
devices with maximum NMSE `7.11e-07`; the `glm5next` regression passed with maximum NMSE `1.11e-07`.
Roundtrip was `OK` in both cases; the Meta backend correctly reported `SKIP`. The device-resident
draft API was not extended to Qwen: the current production helper is limited to GLM5Next, so Qwen
`parallel=2` is covered through ordinary host-driven MTP decode.

### 8.4 Findings Requiring Reproduction

- [x] `seq_cp` with an unfinished rollback plane: the reproducer confirmed loss of `rs_idx`; the
  destination now inherits the source's pending plane; target commit `6fcba3d0c` was created after review.
- [x] Repeated partial `seq_rm` before the next decode and the current hard abort were checked; do not
  add a production fix without a server-flow reproducer and a safe checkpoint fallback.
- [x] Indexer-memory status was checked: both optional indexer contexts are now included in the final
  hybrid status; target commit `b054ef81d` was created after review.
- [x] Quantized indexer K without Hadamard rotation was reproduced and fixed; the Q8 graph now rotates
  keys around cache storage; target commit `d7953e5e8` was created after review.
- [x] Future `output_gate_type` values other than `sigmoid`: the converter now rejects an unsupported
  activation instead of silently producing an invalid GGUF; target commit `4b15fba8a` was created after review.

Do not fix these items blindly. First obtain a minimal reproducer and assess the impact.

For the first item, the reproducer uses hybrid recurrent memory with `src=1`, `dst=0`: the source
decodes 9 tokens, rolls back the last 3, and is copied to the destination without an intervening graph
execution. Before the fix, `rs_idx[src] == 3` but `rs_idx[dst] == 0`, so the test failed. The fix is
one line in `llama_memory_recurrent::seq_cp`: the destination receives the same rollback index. Both
branches then decode the same token in turn and compare against a clean reference; each plane is
consumed independently. CTest passed `4/4`; Qwen35 and Nemotron-H additionally passed with `-ngl all`.
DSV4 has a separate memory implementation and is deliberately unaffected by this change.

The second item reproduces in existing `test_multi_seq_split_replay`: after the first partial
rollback, a deeper second `llama_memory_seq_rm` before graph execution returns `false`. This is the
intentional single-use contract of a pending plane. `common_memory::seq_rm` is explicitly documented
as an aborting wrapper. Replacing the abort with a warning is unsafe because target memory may have
already changed before draft-cache rejection, and continuing would silently produce invalid or
desynchronized state. Rollback planes also cannot be composed without knowing which snapshots the
last decode actually wrote. Therefore, no local fix is needed until a normal server path reproduces
double truncation and supplies a checkpoint from which both memories can be restored atomically.

The third item was confirmed by control-flow analysis: `llama_context` checks `get_status()` before
`apply()`, but `llama_memory_hybrid_context` accounted for only attention and recurrent contexts, and
derived `llama_memory_hybrid_idx_context` did not add the status of its own QSA indexer. This is latent
in the current Qwen4Exp batch path because the attention cache has already prepared the slot, but an
update context can return a compute error, and the memory-context contract requires propagating any
`FAILED_*` status to the caller. A generic three-status combiner was added; base hybrid aggregates
its optional indexer, the Qwen-specific context overrides `get_status()` for its separate indexer,
and defensive assertions use the final virtual status. Combination tests cover `FAILED_PREPARE`,
`FAILED_COMPUTE`, and three `NO_UPDATE` values. The build passed; related CTests passed `5/5`;
Qwen4Exp and GLM5Next passed on CPU and six CUDA devices with maximum NMSE `7.51e-07` and `8.38e-08`
respectively, with roundtrip `OK` everywhere and the Meta backend correctly reporting `SKIP`.

The fourth item was also confirmed, but its cause was refined. The Q8 indexer cache already set
`attn_rot_k = 1` through the generic quantized-cache criterion; the generated Hadamard matrix was not
connected to the Qwen QSA graph. Indexer projection must remain raw through block pooling, norm, and
RoPE, so rotation applies only around storage: `H -> quantized cache -> H`. The second Hadamard can be
performed once after linear pooling; this restores the original representation before order-sensitive
norm/RoPE, preserves existing mathematics, and spreads outliers before quantization. The new Q8
regression requires both operations; F16 requires neither, and `LLAMA_ATTN_ROT_DISABLE=1` retains the
diagnostic raw-cache path. Normal and disabled Qwen4Exp tests passed on CPU and six CUDA devices with
maximum NMSE `7.51e-07`, roundtrip `OK` everywhere; Q8 KV save/load CTest passed `2/2`. Prompt/session
caches saved before the fix with a quantized Qwen indexer must be recreated: old raw Q8 keys are
incompatible with the new rotated storage representation.

The fifth item was reproduced at the converter-contract level. The pinned config of the current
checkpoint contains `hidden_act = "silu"` but separately sets `output_gate_type = "sigmoid"`; the
reference selects `output_gate_type or hidden_act`, while GGUF/runtime Qwen4Exp currently encodes only
sigmoid. Reusing generic `hidden_activation` is invalid because it would change the mathematics of the
current model. The converter now allows only the actually supported sigmoid, with the same fallback to
`hidden_act`, and raises an explicit `ValueError` for silu or an unknown variant. The unit test covers
explicit sigmoid, fallback sigmoid, explicit silu, and missing `output_gate_type`; tests passed `2/2`
in the existing Python environment. New GGUF metadata and a runtime branch should be added only with
a real nonsigmoid checkpoint and a reference parity test.

### 8.5 Excluded Item

- [x] PLE/image `LLAMA_TOKEN_NULL` requires no fix: current `llama-kv-cache.cpp` already preserves
  `ple_image_token_id` for an embeddings-only batch. The finding in one report was a false positive.

## Stage 9. Autofit, Startup, and Runtime OOM

- [-] Do not use autofit for Q4 + shared MTP: borrowing through `ctx_other` is not included in joint
  measurement; this combination uses the verified manual split. The target Q8 + self-contained
  configuration uses corrected autofit and passes `ctx=524288`, `parallel=2`.
- [x] Treat startup graph-reserve OOM and runtime TOP_K/CUB OOM as separate problems: the manual
  configuration sees only a startup pipeline-parallel reserve OOM with successful fallback; the
  first decode and repeated concurrent requests complete without a late OOM.
- [x] Check the effects of `--ubatch-size`, pipeline parallelism, and GPU split: Q4+shared with the
  manual split requires `ubatch=128`, while Q8+self-contained with autofit and `fit-target=3072`
  starts reliably with `ubatch=512`; the long prompt reached `689.49 tok/s` cold and
  `837.58 tok/s` warm.
- [x] Record peak VRAM on every GPU at load and first decode: for Q4+shared, post-load was
  `43219,47749,39963,11315,43821,23431 MiB`, and after the first concurrent decode it was
  `43591,48205,40335,11409,44193,23585 MiB`; for Q8+self-contained, sampled prefill VRAM remained
  constant at `377,45087,45269,16103,45393,19795 MiB`, and post-decode resident VRAM was
  `377,45293,45475,16161,45589,19877 MiB` on physical GPUs 0-5.
- [x] Verify that #28671 removes late `cuMemCreate` OOM in practice: two concurrent 1024-token
  decodes and a separate 4096-token decode completed without `cuMemCreate`, CUB, or runtime OOM;
  server health remained `ok`.
- [x] Fix infinite Q8 KV autofit: eliminate unsigned `n_part` underflow, add runtime partition
  invariant checks, and synchronize self-contained MTP estimation with the candidate target split.
- [x] Check whether startup OOM disappears after removing the indexer V-cache: the Q8 KV
  configuration with split `15,14,14,4,1,1` reserves target and MTP graphs without OOM or pipeline fallback.
- [-] PR #28569 is not needed now: Q8+self-contained fits through standard autofit, while Q4+shared
  has a verified manual split.
- [-] Do not include PR #28118 at this stage: current runtime stress passed, and changing recurrent
  checkpoints requires a separate performance/correctness study.

The former `-ctk q8_0 -ctv q8_0` limitation is removed by commit `c5f079ab3`: autofit takes
`16.75-16.93 s`, and full startup takes `49.33-49.57 s`. Until deployment, the working preset for
the installed binary should still omit KV quantization and loads in `64.39 s`.

Completion criterion: autofit finishes in a bounded and measured time, or a documented stable manual
configuration is used; the server does not fail during long decode.

## Stage 10. Final Validation Matrix

| Target | Draft | Context | Parallel | Text | Vision | Concurrent | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Q4 | off | short | 1 | [x] | [x] | n/a | [x] |
| Q4 | MTP Q8 | short | 1 | [x] | [x] target-only fallback | n/a | [x] `367bea7f1` |
| Q4 | MTP Q8 | 131k+ | 1 | [x] | [x] target-only fallback | n/a | [x] |
| Q8 | off | short | 1 | [x] | [x] | n/a | [x] |
| Q8 | MTP Q8 | short | 1 | [x] | [x] target-only fallback | n/a | [x] |
| Q8 | MTP Q8 | 131k+ | 1 | [x] | [x] target-only fallback | n/a | [x] |
| Q4 | off | 524288 total | 2 | [x] | [x] | [x] | [x] |
| Q4 | MTP Q8 | 524288 total | 2 | [x] | [x] target-only per-slot | [x] | [x] |
| Q8 | off | 524288 total | 2 | [x] | [x] | [x] | [x] |
| Q8 | MTP Q8 | 524288 total | 2 | [x] | [x] target-only per-slot | [x] | [x] |

Final required checks:

- [x] clean build;
- [x] existing unit/backend tests: 63/64 CTest targets completed, while the full
  `test-backend-ops` was stopped by the overall 1500-s timeout without an operation failure; all
  affected KPOOL, TOP_K/indexer, QSA, architecture, and recurrent subsets were rerun separately;
- [x] deterministic greedy repetitions: Q4/Q8 short MTP off/on and Q4/Q8 148k are byte-identical;
- [x] parallel isolation: Q4/Q8 target-only and MTP, including concurrent vision+text and both
  post-vision slots, are complete;
- [x] prompt-cache save/restore: target state restores correctly, but the file format does not save
  draft/spec state and performs a full MTP prefill; the user rejected a file-format change;
- [x] MTP accept/reject/rollback: synthetic CTest and runtime text -> vision target-only -> text MTP
  transitions passed;
- [x] text and multimodal: Q4/Q8 short and 148k are complete; Qwen vision uses target-only fallback;
- [x] at least one long run: Q4 148k text and full multimodal prefill;
- [x] no startup or runtime OOM: single-slot and `parallel=2` Q4/Q8 completed;
- [x] documented rollback switches for experimental optimizations.

### Clean Build and Backend Gate, 2026-09-11

A completely new `build-final-qwen38` build was created from `c5f079ab3`: Release, CUDA, Flash
Attention, CUDA graphs, architectures `86-real;89-real`, server, and tests. Configuration with CUDA
toolkit `13.0.88` and the full build completed successfully.

The sequential full CTest completed 63 of 64 targets. The only unfinished target,
`test-backend-ops`, was stopped by its own timeout after exactly `1500.61 s`, not by a crash or OOM;
it had tested CUDA0-CUDA3 by then. On each GPU, the same `KPOOL_EXPAND(kpool=8,...)` case was marked
failed because of `NaN at index 768`, with the same `-nan` on both sides of the comparison. The
operation output itself is I32 and cannot contain NaN. The cause was F32 sentinel tensors in the
shared test harness: `test_kpool_expand` did not initialize unnamed tensors, so identical
uninitialized bytes could be interpreted as NaN.

The minimal fix calls `init_tensor_uniform(t)` for the remaining tensors in
`test_kpool_expand::initialize_tensors()`. Results after rebuilding:

- `KPOOL_EXPAND`: `2/2` on each of six CUDA GPUs, `7/7 backends passed`;
- `TOP_K,KPOOL_EXPAND,LIGHTNING_INDEXER`: `682/682` on each of six CUDA GPUs,
  `7/7 backends passed`;
- `TOPK_QSA,QSA_BLOCK_SELECT`: `10/10` on each of six CUDA GPUs (`4/4` + `6/6`),
  `7/7 backends passed`.

The original full run also crossed an existing F16 tolerance boundary once in `ADD` on CUDA1 and
once in `ADD_ADD` on CUDA2. Isolated reruns without kernel changes first produced `5/7` and then
`6/7 backends passed`, so the failure set is unstable. This is a separate tolerance flake in the
generic backend test and is unrelated to the deterministic KPOOL sentinel fix; no change for it was
included in that commit. The final runtime matrix was later completed in full.

### Q4 Short Text/Vision and Qwen Multimodal MTP Fallback, 2026-09-11

Profile: Q4 target, `ctx=8192`, `parallel=1`, `batch=2048`, `ubatch=512`, `fit-target=3072`,
`temperature=0`, `seed=1234`, six GPUs ordered `2,1,0,5,4,3`. Vision uses `mmproj-F16.gguf` and the
standard `tools/mtmd/test-1.jpeg` image headed `MEN WALK ON MOON`.

Without MTP, two identical text requests produced byte-identical reasoning/final, the correct result
`472`, and `63.24/64.11 tok/s`. Vision correctly read the headline and identified Apollo 11: prefill
`399.85 tok/s`, decode `59.14 tok/s`.

With shared Q8 MTP, text again repeated byte-for-byte and returned the correct `472`; acceptance was
`270/304 = 0.88816`, decode `103.12/109.08 tok/s`. However, the original vision test consistently
returned HTTP 500: `ctx_dft sequence 0 cannot catch up from pos=46 to pos=66`. Commit `432d41f03`,
which introduced a multimodal MTP fast path for GLM5Next, enabled it for every MTP architecture.
Qwen image batches use four M-RoPE position planes, while this draft catch-up carries only an ordinary
one-dimensional position; continuing such decode is mathematically invalid.

Fix `367bea7f1` leaves the media fast path enabled only for draft architecture `glm5next`. For a Qwen
task containing `LLAMA_TOKEN_NULL`, MTP is temporarily disabled on that slot, draft memory is cleared,
and target continues ordinary multimodal decode. The next text task re-enables MTP and, when needed,
fully reprocesses the prompt instead of reusing an incompatible media cache.

After the fix, two consecutive vision -> text cycles in one server process completed without HTTP 500:

- vision twice returned the correct `MEN WALK ON MOON`/Apollo 11 in target-only mode, with decode
  `59.68` and `56.96 tok/s`;
- the following text tasks used MTP again: acceptance `253/282 = 0.89716` and `39/44 = 0.88636`;
  answers `472` and `323` were correct;
- related CTests `test-arg-parser`, `test-model-resolution`, `test-mtmd-c-api`, and `test-mtmd-impl`
  passed `4/4`.

Full speculative MTP for Qwen vision would require transferring all four M-RoPE planes and separately
validating hidden/token pairing. Until then, target-only fallback is the correct working behavior;
the GLM5Next fast path remains enabled.

### Q4 + Shared Q8 MTP, Structured 148k Text/Vision, 2026-09-11

Profile: Q4 target, shared Q8 MTP, `ctx=262144`, `parallel=1`, `batch=512`, `ubatch=128`, `fit=off`,
split `27,24,27,24,8,6`, Q8 target KV, `spec-draft-n-max=3`, `temperature=0`, `seed=1234`. This uses
the same KV capacity per slot as the working `ctx=524288`, `parallel=2` configuration. The server with
`mmproj-F16.gguf` loaded in `23.59 s` without startup OOM. A control VRAM snapshot during long prefill
showed physical GPU0-5 usage of `9071,21473,19053,5495,24469,21081 MiB`, leaving at least
`3044 MiB` free on GPU5.

The structured text prompt contains 4000 numbered records and three unique keys near the beginning,
middle, and end. The actual chat prompt is `148159` tokens. The first full pass took `342.39 s`, with
prefill `432.72 tok/s`, decode `41.97 tok/s`, acceptance `154/186 = 0.82796`, and exact answer
`731942 804271 459806`. An identical repeat produced byte-identical reasoning/final, reused
`148155/148159`, had the same acceptance, and completed in `5.76 s` (`43.34 tok/s` decode).

The long-vision prompt places `tools/mtmd/test-1.jpeg` after the same log. The Qwen fallback correctly
disabled MTP for the media task and independently reprocessed `148416` of `148458` prompt tokens in
`314.70 s`, with prefill `471.61 tok/s` and target-only decode `32.02 tok/s`. The first run hit only
the configured `max_tokens=256`: reasoning already contained all three correct keys and
`MEN WALK ON MOON`, but final output had not begun. A retry with `max_tokens=512` reused
`148454/148458` and returned exact final `731942 804271 459806 | MEN WALK ON MOON` in `16.05 s`,
with decode `32.62 tok/s`.

The diagnostic `non-consecutive token position` messages during media decode represent Qwen's four
M-RoPE position planes and were not accompanied by an error. The following short text request saw the
expected incompatible target/draft cache after target-only media, fully reprocessed its prompt,
re-enabled MTP, and returned the correct `472`: acceptance `76/84 = 0.90476`, decode
`107.38 tok/s`. No HTTP 500, `cannot catch up`, startup/runtime OOM, or state corruption was observed.

### Q8 Short Text/Vision Without MTP and With Self-Contained Q8 MTP, 2026-09-11

Both profiles use a Q8 target, `ctx=8192`, `parallel=1`, `batch=2048`, `ubatch=512`, Q8 target KV,
`load-mode=none`, `lazy-mode=auto`, `temperature=0`, `seed=1234`, and six GPUs ordered
`2,1,0,5,4,3`. The MTP profile adds a self-contained Q8 draft and `spec-draft-n-max=3`. These short
commands contained both `-fit on` and explicit `-ngl all`/`-ngld all`; the server warned
`n_gpu_layers already set by user to -2, abort`, so they actually used full manual offload rather
than autofit. This does not change short-request correctness or timing; a separate real autofit test
appears below without conflicting `-ngl`/`-ngld` options.

The target-only server loaded in `58.81 s`; two identical text requests matched byte-for-byte and
returned the correct `472`, with decode `55.59/58.16 tok/s`. Vision correctly read
`MEN WALK ON MOON` and identified Apollo 11: prefill `464.23 tok/s`, decode `54.28 tok/s`. At least
`6105 MiB` remained free after load, and health stayed `ok`.

The self-contained MTP server with full manual offload loaded in `40.70 s` from a warm storage cache.
Two identical text requests again matched byte-for-byte and returned `472`: acceptance
`76/84 = 0.90476`, decode `96.13/105.72 tok/s`, approximately `1.73-1.82x` faster than target-only
decode on this short prompt. Vision correctly switched to target-only fallback and returned the same
headline and Apollo 11: prefill `455.69 tok/s`, decode `53.52 tok/s`. The following text request used
MTP again and correctly returned `323`, with acceptance `48/54 = 0.88889`, decode `98.20 tok/s`.

The final Q8+MTP VRAM snapshot on physical GPUs 0-5 was
`26625,27427,27455,14015,28255,14017 MiB`, leaving at least `6039 MiB` free. No startup/runtime OOM,
HTTP 500, or target/draft state errors occurred; final health was `ok`.

### Q8 + Self-Contained Q8 MTP, Structured 148k Text/Vision, 2026-09-11

Profile: Q8 target, self-contained Q8 MTP, `ctx=262144`, `parallel=1`, `batch=2048`, `ubatch=512`,
Q8 target KV, `fit=on`, `fit-target=3072`, no `-ngl`/`-ngld`, `load-mode=none`, `lazy-mode=auto`,
`spec-draft-n-max=3`, `temperature=0`, `seed=1234`. Real autofit completed, and the full server loaded
in `46.45-49.20 s`. An attempt with explicit `-ngl all` was immediately rejected by the fitter and
was not used for results. After fitting, physical GPU3 remained almost empty; the minimum post-load
free memory on used GPUs was `3456 MiB`.

The same structured prompt as in the Q4 test produced `148159` prompt tokens and exact answer
`731942 804271 459806`. Two independent cold runs after server startup reached prefill `604.75` and
`609.04 tok/s`; the latter had decode `42.05 tok/s`, acceptance `139/154 = 0.90260`. An identical
repeat reused `148155/148159` tokens, produced byte-identical reasoning/final, had the same acceptance,
and completed in `4.91 s` (`42.52 tok/s` decode). Compared with long Q4, the first Q8 prefill was
approximately `39.7%` faster, and acceptance increased from `0.82796` to `0.90260`.

The long-vision prompt contains `148458` tokens. The architecture-gated Qwen fallback temporarily
disabled MTP and processed `148416` new tokens in `216.43 s`: prefill `685.73 tok/s`, target-only
decode `27.06 tok/s`. Final output was exact: `731942 804271 459806 | MEN WALK ON MOON`. The following
text request forced the expected full reprocessing of the incompatible media/draft cache, re-enabled
MTP, and returned `323`: acceptance `48/54 = 0.88889`, decode `88.64 tok/s`.

The final VRAM snapshot on physical GPUs 0-5 was `1727,45375,45377,285,45293,21443 MiB`, leaving at
least `2682 MiB` free. Health was `ok`; no startup/runtime OOM, HTTP 500, `cannot catch up`, or
target/draft state corruption occurred.

### Q4, ctx=524288, parallel=2 Without MTP and With Shared Q8 MTP, 2026-09-11

Both profiles use a Q4 target, two slots of `262144` tokens, `batch=512`, `ubatch=128`, Q8 target KV,
`fit=off`, split `27,24,27,24,8,6`, `load-mode=none`, `lazy-mode=auto`, `temperature=0`, `seed=1234`.
The MTP profile adds a shared Q8 draft and `spec-draft-n-max=3`.

The target-only server loaded in `37.72 s`. The first concurrent stress with two 128-number ranges
intentionally reached `max_tokens=1024` in both slots: decode was `48.57/48.32 tok/s`, with no runtime
error or reasoning mix. For strict final validation, ranges were shortened to 32 numbers: both were
fully correct, contained only their own `ALPHA`/`BETA`, and omitted the other marker; decode was
`48.35/55.21 tok/s`. Concurrent vision+text correctly returned `MEN WALK ON MOON`/Apollo 11 and
independent result `519`: vision prefill `307.43 tok/s`, decode `50.00 tok/s`, neighboring text decode
`43.11 tok/s`. Health was `ok`, with at least `2094 MiB` of VRAM free.

Q4 + shared MTP loaded in `24.62 s`. The same strict concurrent pair returned complete isolated
ranges: acceptance `144/160 = 0.90000` and `291/324 = 0.89815`, decode `49.93/68.19 tok/s`. During
concurrent vision+text, the media slot correctly switched to target-only and returned the headline
and Apollo 11, while the neighboring text slot continued using MTP, returned `519`, with acceptance
`59/66 = 0.89394` and decode `50.51 tok/s`. The next concurrent text pair confirmed MTP recovery in
both slots: correct `323`/`408`, acceptance `48/54 = 0.88889` and `59/66 = 0.89394`, decode
`69.91/81.89 tok/s`.

The final Q4+MTP VRAM snapshot on physical GPUs 0-5 was
`10403,22747,20019,5549,25785,22349 MiB`, leaving at least `1776 MiB` free. Health was `ok`; no
startup/runtime OOM, HTTP 500, cross-slot marker leakage, or target/draft state corruption occurred.
Multimodal MTP fallback was confirmed to be per-slot rather than a global draft disable.

### Q8, ctx=524288, parallel=2 Without MTP and With Self-Contained Q8 MTP, 2026-09-11

Both profiles use a Q8 target, two slots of `262144` tokens, `batch=2048`, `ubatch=512`, Q8 target KV,
real `fit=on`, `fit-target=3072`, no explicit `-ngl`/`-ngld`, `load-mode=none`, `lazy-mode=auto`,
`temperature=0`, `seed=1234`. The MTP profile adds a self-contained Q8 draft and
`spec-draft-n-max=3`.

Target-only autofit completed, and the server loaded in `53.09 s`. The strict concurrent text pair
returned complete ranges containing only their respective `ALPHA`/`BETA` markers, with no cross-slot
leakage; decode was `37.42/43.02 tok/s`. Concurrent vision+text correctly returned
`MEN WALK ON MOON`/Apollo 11 and independent result `519`: vision prefill `325.33 tok/s`, decode
`40.01 tok/s`, neighboring text decode `35.26 tok/s`. Health was `ok`, with at least `3050 MiB` free.

Q8 + self-contained MTP autofit completed, and the server loaded in `65.76 s`. The same concurrent
text pair returned exact isolated ranges: acceptance `147/155 = 0.94839` and `318/351 = 0.90598`,
decode `54.59/63.20 tok/s`, compared with target-only `37.42/43.02 tok/s`. During concurrent
vision+text, the media slot correctly used target-only fallback and returned the headline/Apollo 11,
while the neighboring text slot continued using MTP, returned `519`, with acceptance
`59/66 = 0.89394` and decode `43.74 tok/s`. The following concurrent text pair correctly returned
`323`/`408` and confirmed MTP in both slots: acceptance `48/54 = 0.88889` and `59/66 = 0.89394`,
decode `61.92/59.89 tok/s`.

The final Q8+MTP VRAM snapshot on physical GPUs 0-5 was
`1733,45171,44821,14091,45411,13963 MiB`, leaving at least `3098 MiB` free. Health was `ok`; no
startup/runtime OOM, HTTP 500, cross-slot marker leakage, or target/draft state corruption occurred.
All ten rows of the final runtime matrix are complete.

### File-Based Prompt-Cache Save/Restore With MTP, 2026-09-11

Validation used a Q8 target + self-contained Q8 MTP, `ctx=8192`, `parallel=2`, Q8 KV,
`batch=2048`, `ubatch=512`, full GPU offload, and `spec-draft-n-max=3`. The server was started with a
dedicated `--slot-save-path`; slot 0 processed a `1798`-token prompt, after which its state was saved
through `POST /slots/0?action=save` and restored into empty slot 1 through
`POST /slots/1?action=restore`.

The API successfully wrote and read `1813` tokens and `144763540` bytes; output after restore matched,
MTP retained acceptance `9/15 = 0.60000`, and there was no state error. However, the first request
after restore had `cache_n=0`, `prompt_n=1798`: the prompt was processed in full. The next request in
the same slot used the RAM cache (`cache_n=1794`, `prompt_n=4`) with the same output and acceptance.

The cause was confirmed in server slot API code: file save calls
`llama_state_seq_save_file(ctx_tgt, ...)` for target only. The internal RAM prompt cache separately
saves target, draft, and speculative state, so ordinary reuse works. File restore remains correct,
but draft/spec state is absent and MTP safely performs a full prefill. The user chose not to extend
the file format; this is a recorded limitation, not an unfinished fix.

## Results by Stage

Fill this table after every stage using an identical prompt set.

| Stage | Git SHA / diff | Target | Context | Parallel | MTP | Acceptance | tok/s | VRAM / peak | Tests | Result |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| Baseline | `d1a92352` | historical source | n/a | n/a | n/a | n/a | incomparable old measurements | not measured | build/architecture baseline | do not use for strict A/B; current baseline is provided by the final rows |
| GDN | `36f1df783` | Q4/Q8 | 4096 | 1 | off/on | n/a: short answer | 48.8-69.2 generation tok/s | not measured | CPU/CUDA architecture OK; CUDA TOPK_QSA 4/4 | committed after review |
| #28671 | `bc495e524` (`ff2b436` port) | CUDA TOP_K | 4096-131072 width | n/a | n/a | n/a | 2.24-2.73x kernel speedup for width >= 8192 | wide tests without OOM | TOP_K 525/525 on both paths; architecture OK; QSA 4/4 | committed after review; tied cells fixed in stage 4 |
| #28213 | `6593b0abb` (`beed2f7` port) | Q4/Q8 | long prompt, ctx 32768 | 1 | off/on smoke | not measured | Q4 off: 53.5 masked / 55.6 gather; Q4+MTP 64.8 | Q8 full manual offload OOM; partial offload OK | CPU/CUDA architecture OK; Q4 A/B; Q8 smoke; MTP smoke | committed after review; MTP verification still does not use gather |
| QSA correctness | `6872b63c1` source / `96990f5a7` stable | Q4 + synthetic | 2047-2051, 32771; Q4 ctx 32768 | 1 | off | n/a | Q4 gather/masked 54.6/54.6 smoke | no new OOM | CUDA block select 6/6 on 6 GPUs, repeats 20/20, architecture OK, Q4 A/B | committed after review; direct HF logits parity requires the source checkpoint |
| no indexer V-cache | `d1946ac89` source / `9f698750f` stable | Q4 | 524288 total | 2 | off | n/a | n/a | Q8 indexer 816 MiB, V 0 MiB | architecture/QSA OK; test 5 later fixed in `2ecc7b194` | committed after review |
| stable branch integration | `36279d50d..9f698750f` | Q4 | 4096-524288 | 1/2 | off | n/a | n/a | no new OOM | per-stage build/tests | five correctness fixes ported separately |
| Qwen self-contained MTP | `8ac6982b1` | Q4 + MTP Q8 | 4096 | 1 | on | 0.73214, 41/56 | 72.18-78.23 tok/s | Q8 target manual split OOM | build; architecture CPU/CUDA OK; real generation OK | committed after review |
| Qwen shared MTP borrowing | `98a58b551` | Q4 + shared MTP Q8 | 4096 | 1 | on | 0.73214, 41/56; autofit smoke 0.75472, 40/53 | 67.07 tok/s A/B; 93.36 tok/s autofit smoke | fitter measures target without draft after expected warning | build; architecture CPU/CUDA OK; self-contained A/B parity; autofit generation OK | committed after review; KV memory is not shared |
| quantized device state copy | `2ecc7b194` | Q4 + Q8 KV; qwen4exp dummy | 4096 / 256 | 1 | off | n/a | n/a | device snapshot now stores every quant block | CPU/CUDA Q8 KV regression; real Q4 tests 1-8 | committed after review; generic fix for quantized state views |
| GLM5Next reserve sequences/streams | `8a8c83913` | glm5next dummy | 256 | 1/2 synthetic | off | n/a | n/a | CUDA reserve without runtime resize warning | isolated CPU/CUDA tests 1-8; full CTest and CUDA matrix; architecture on CPU + 6 GPUs | committed; runtime memory layout unchanged |
| MTP parallel fix | `874703118` + `803b76c92` + `ccdf0ae2c` | Q4 + shared MTP Q8 | 8192 / 524288 | 2 | on | 0.582-0.728 in concurrent runs | 34 tok/s per slot in the first full concurrent decode; up to 79-81 tok/s in short repeats | 524k post-load: 43219,47749,39963,11315,43821,23431 MiB | graphs on/off; target-only control; 5 concurrent rounds; prompt-cache reuse | no state errors or semantic transfer; multi-hour soak excluded from the fix plan |
| 524k manual placement | runtime config | Q4 + shared MTP Q8 | 524288 | 2 | on | 0.593/0.655 in first concurrent decode | 34.06/34.77 tok/s | split `27,24,27,24,8,6`; at least 304 MiB free after decode | 273/284-token first decode + repeated requests | works, but VRAM headroom is very small and pipeline parallelism falls back |
| MTP QSA | `65157dfd4` | Q4 + shared MTP Q8 | 2041-148159 / 524288 total | 1/2 | QSA + Top-K share | 0.54955-0.63492 depth grid; 0.73828 A/B; 0.529-0.600 concurrent; 0.82796 at 148k | 31.92-58.92 depth grid; 63.69 A/B | full config server: 22936,25436,20508,5568,23240,10334 MiB | build; CPU/CUDA architecture; TOPK_QSA 4/4; threshold; 32k/64k/128k; structured 148k; fallbacks; parallel runtime | committed; actual 262k prompt deliberately excluded |
| Autofit/runtime OOM | runtime config | Q8 + self-contained MTP Q8 | 524288 total | 2 | on, depth 3 | 0.678/0.689 concurrent; 0.721 long decode | prefill 689.49 cold / 837.58 warm; decode 30.19/30.51 concurrent, 60.02 single | physical GPU post-decode: 377,45293,45475,16161,45589,19877 MiB | 42.8k long prompt x2; concurrent 2x1024; single 4096; health OK | working preset loads in 64.39 s; no runtime OOM; do not use >10-min Q8 KV fit |
| Q8 KV autofit fix | `c5f079ab3` | Q8 + self-contained MTP Q8, Q8 target KV | 524288 total | 2 | on, depth 3 | 0.6458 long; 0.7183/0.7087 concurrent | fit 16.75-16.93 s; startup 49.33-49.57 s; prefill 844.23; decode 40.88 single, 33.35/32.61 concurrent | split `15,14,14,4,1,1`; physical post-load 1731,45013,44661,13763,44563,14739 MiB | build; CTest 6/6; short, 42.9k, and parallel=2 runtime; health OK | underflow and stale MTP placement fixed; committed after review |
| Final clean backend gate | `3bb01a85f` | synthetic backend ops | n/a | n/a | n/a | n/a | n/a | n/a | clean CUDA build; CTest 63/64 targets before timeout; KPOOL 2/2, TOP_K/KPOOL/INDEXER 682/682, QSA 10/10 on 6 GPUs | KPOOL false NaN fixed; committed after review |
| Q4 short final | `367bea7f1` | Q4 + shared MTP Q8 | 8192 | 1 | off/on, depth 3 | 0.88636-0.89716 | text 63.24-64.11 off, 103.12-109.08 on; vision 56.96-59.68 target-only | no OOM | deterministic text; vision x2; vision->text x2; CTest 4/4 | Qwen vision HTTP 500 fixed; committed after review |
| Q4 148k final | `367bea7f1` | Q4 + shared MTP Q8 | 262144 | 1 | text on, vision target-only; depth 3 | text 0.82796; post-vision 0.90476 | text prefill 432.72/decode 41.97; vision prefill 471.61/decode 32.02 | at least 3044 MiB free, no OOM | exact 3-marker retrieval; deterministic repeat; 148k vision; cache reuse; post-vision MTP | single-slot long text/vision complete; committed after review |
| Q8 short final | `c5f079ab3` + `367bea7f1` | Q8 + self-contained MTP Q8 | 8192 | 1 | off/on, depth 3; vision target-only | 0.90476 text; 0.88889 post-vision | text 55.59-58.16 off, 96.13-105.72 on; vision 53.52-54.28 | at least 6039 MiB free, no OOM | deterministic text off/on; vision off/on; post-vision MTP; health OK | Q8 short with manual full offload complete; plan commit `daebbe525`; autofit description corrected later |
| Q8 148k final | `c5f079ab3` + `367bea7f1` | Q8 + self-contained MTP Q8 | 262144 | 1 | text on, vision target-only; depth 3 | text 0.90260; post-vision 0.88889 | text prefill 604.75-609.04/decode 42.05; vision prefill 685.73/decode 27.06 | at least 2682 MiB free, no OOM | real autofit; exact 3-marker retrieval; deterministic repeat; cache reuse; 148k vision; post-vision MTP | Q8 single-slot long complete; plan commit `4f1ccff75` |
| Q4 parallel final | `367bea7f1` | Q4 + shared MTP Q8 | 524288 total | 2 | off/on, depth 3; vision target-only per-slot | 0.89815-0.90000 concurrent; 0.89394 mixed; 0.88889-0.89394 post-vision | text 48.35-55.21 off, 49.93-68.19 on; mixed vision 39.42/text 50.51 | at least 1776 MiB free, no OOM | isolated ALPHA/BETA; 2x1024 stress; concurrent vision+text; both post-vision slots; health OK | Q4 parallel off/on complete; plan commit `693ccdcbb` |
| Q8 parallel final | `c5f079ab3` + `367bea7f1` | Q8 + self-contained MTP Q8 | 524288 total | 2 | off/on, depth 3; vision target-only per-slot | 0.90598-0.94839 concurrent; 0.89394 mixed; 0.88889-0.89394 post-vision | text 37.42-43.02 off, 54.59-63.20 on; mixed vision 31.61/text 43.74 | at least 3098 MiB free, no OOM | real autofit off/on; isolated ALPHA/BETA; concurrent vision+text; both post-vision slots; health OK | complete runtime matrix; plan commit `4cae0590f` |
| MTP file slot cache | runtime behavior | Q8 + self-contained MTP Q8 | 8192 | 2 | on, depth 3 | 0.60000 before/after restore | full restored prompt 1158.81 tok/s; RAM reuse processes 4 tokens | no OOM | save slot 0, restore slot 1, repeat; identical output | target restore is correct; draft/spec are not saved, so the first MTP prefill is full; format change rejected |
| wildcard recurrent seq_rm | `2129b17a9` | Qwen35 / Nemotron-H / DeepSeek4 dummy | 64-512 | 2 | off | n/a | n/a | n/a | recurrent rollback CTest 4/4; manual CPU/CUDA; save/load 2/2 | committed after review; fixed recurrent, base KV, and DSV4 no-op |
| converter EOS list | `6a413fbcc` | converter unit test | n/a | n/a | n/a | n/a | n/a | n/a | scalar/list EOS 2/2 | committed after review; primary EOS is the first list item |
| Qwen MTP test coverage | `9b374c470` | synthetic Qwen4Exp/GLM5Next | 256 | 1/2 | self-contained/shared | n/a | n/a | n/a | CPU + 6 CUDA; roundtrip; QSA reuse; rollback | committed after review |
| recurrent seq_cp rollback plane | `6fcba3d0c` | Qwen35 / Nemotron-H dummy | 64 | 2 | off | n/a | n/a | n/a | reproducer before fix; CTest 4/4; CUDA 2/2; architecture regressions | committed after review |
| GDN short-batch rollback | `26abcbf7e` | Qwen3.8 Q4 + synthetic recurrent models | 64-512 | 1/2 | off | n/a | n/a | n/a | real Q4 6-GPU; synthetic Qwen4Exp; recurrent CTest 4/4 | input state saved in plane `n_seq_tokens`; all replay comparisons `max diff 0` |

## Work Log

| Date | Stage | Status | Work completed | Next step |
|---|---|---|---|---|
| 2026-09-10 | Planning | `[x]` | Defined fix order, dependencies, and test gates; selected the primary working Git directory | Stage 0: create the working branch and capture a baseline |
| 2026-09-10 | Planning | `[x]` | Added mandatory transfer to the stable branch with existing MTP/parallel fixes after the V-cache fix | In stage 5.5, record branch and base SHA, then repeat the baseline on the new base |
| 2026-09-10 | Stage 0 | `[x]` | Created `work/qwen38-fixes`; recorded HEAD, CUDA/driver/CCCL, and active TOP_K fallback; stages 9-10 later added reproducible Q4/Q8 short, 148k, and parallel=2 baselines | Use the final matrix as the current branch baseline |
| 2026-09-10 | Stage 1: GDN | `[x]` | Minimally ported the qwen4exp portion of PR #28068; CPU/CUDA tests and Q4/Q8 smoke with MTP off/on passed; created `36f1df783` after review | Stage 2: PR #28671 |
| 2026-09-10 | Stage 2: #28671 | `[x]` | Ported `ff2b436` exactly; CUDA build passed; TOP_K 525/525 on radix-select and argsort; kernel A/B showed 2.24-2.73x speedup at widths 8192-131072; created `bc495e524` after review | Stage 3: PR #28213 |
| 2026-09-10 | Stage 3: #28213 | `[x]` | Ported `beed2f7` while preserving local MTP; CPU/CUDA build and architecture tests passed; Q4 masked/gather A/B gave +3.9%; Q8 and Q4+MTP smoke passed; created `6593b0abb` after review | Stage 4: QSA block selection |
| 2026-09-10 | Stage 4: QSA correctness | `[x]` | Implemented TOP_K over complete blocks and a separate tail without epsilon; masked/gather tests 6/6 on 6 GPUs, 20/20 repeats; source commit `6872b63c1`, target commit `96990f5a7` | Complete and integrated into the stable branch |
| 2026-09-10 | Stage 5.5: stable integration | `[x]` | Ported #28671, #28213, QSA correctness, and no-V-cache individually into `my_build_qwen4next`; GDN was already in the base | Adapt Qwen self-contained MTP to the stable API |
| 2026-09-10 | Qwen self-contained MTP | `[x]` | Added NextN tensors, draft-only loader, dense `graph_mtp`, plain MTP KV cache, and 2D backend hidden output; Q4 acceptance 0.73214; created `8ac6982b1` after review | Implement borrowing in a separate commit |
| 2026-09-10 | Qwen shared MTP borrowing | `[x]` | Added embedding/output borrowing through `ctx_other`; KV sharing remains Gemma4 Assistant-only; shared and self-contained both produced `41/56`; autofit smoke passed; created `98a58b551` after review | Fix baseline test 5 |
| 2026-09-10 | Stage 6: state copy | `[x]` | Found a `bytes/type_size` error for block-quantized views; added block-aware calculation and Q8 KV CTest; CPU/CUDA dummy and real Q4 passed tests 1-8; created `2ecc7b194` | Fix the discovered GLM5Next reserve failure |
| 2026-09-10 | GLM5Next reserve | `[x]` | Separated logical sequence count from unified KV stream count in the synthetic context; isolated CPU/CUDA, full save/load matrices, and architecture tests passed; created `8a8c83913` | Check existing MTP/parallel fixes on Qwen3.8 |
| 2026-09-10 | Stage 6: MTP parallel | `[x]` | Checked fixes, target-only control, graphs on/off, fixed slots, prompt-cache reuse, and concurrent runtime; automated Qwen MTP and recurrent rollback regressions; GDN short-batch fix `26abcbf7e` passed real-Q4 replay | Multi-hour soak left as operational validation |
| 2026-09-10 | Stage 9: autofit 524k | `[-]` | Adopted a verified manual split for Q4+shared; fixed Q8+self-contained autofit in `c5f079ab3`, passing `ctx=524288`, `parallel=2` | Do not extend the fitter for borrowing without a separate generic design |
| 2026-09-10 | Stage 7: MTP QSA | `[x]` | Added loader fallback ratio=4, hybrid K-only MTP indexer, stream-local reuse of 512 block IDs, selection-ready guard, and two runtime kill switches; direct 21.4k A/B, 32k/64k/128k depth grid, structured 148k, and full parallel runtime passed; created `65157dfd4` | Complete; actual 262k prompt deliberately excluded |
| 2026-09-10 | Stage 8.1: wildcard seq_rm | `[x]` | Fixed early rejection of any negative ID, base-KV assertion, stale recurrent tails after full removal, and DSV4 empty-range no-op; the new test reproduced errors before the fixes and passes on three architectures CPU/CUDA; created `2129b17a9` after review | Stage 8.2: EOS list in converter |
| 2026-09-11 | Stage 8.2: EOS list | `[x]` | Converter selects the first list item as primary EOS; scalar/list unit test passes; created target commit `6a413fbcc` | Stage 8.3: automated MTP coverage |
| 2026-09-11 | Stage 8.3: MTP tests | `[x]` | Added self-contained/shared borrowing, host-driven parallel=2, QSA selection reuse, and rollback of two draft tokens; Qwen4Exp and GLM5Next passed CPU + 6 CUDA; created target commit `9b374c470` | Stage 8.4: first reproduce `seq_cp` with an unfinished rollback plane |
| 2026-09-11 | Stage 8.4: seq_cp rollback | `[x]` | Reproducer confirmed pending-plane loss on `seq_cp(1,0)`; added `rs_idx` transfer; both branches match the reference; CTest 4/4 and CUDA Qwen35/Nemotron-H passed; created target commit `6fcba3d0c` | Reproduce repeated partial `seq_rm` and the wrapper's hard abort |
| 2026-09-11 | Stage 8.4: repeated seq_rm | `[x]` | Confirmed intentional rejection of a second pending partial rollback; suppressing abort is unsafe because target/draft can desynchronize; no runtime server reproducer | Do not change production code; check indexer-memory status in hybrid context |
| 2026-09-11 | Stage 8.4: indexer status | `[x]` | Confirmed base hybrid and Qwen-specific contexts hid the status of their indexers; added third-status aggregation and virtual-status assertions; CTest 5/5 and Qwen4Exp/GLM5Next CPU + 6 CUDA passed; created `b054ef81d` | Check quantized indexer K without Hadamard rotation |
| 2026-09-11 | Stage 8.4: indexer Hadamard | `[x]` | Refined the cause: Q8 cache enabled rotation but the QSA graph did not use the matrix; added `H -> cache -> H` before norm/RoPE plus F16/Q8/disable coverage; Qwen4Exp CPU + 6 CUDA and Q8 state test passed; created `d7953e5e8` | Check future `output_gate_type != sigmoid` |
| 2026-09-11 | Stage 8.4: output gate | `[x]` | Confirmed reference semantics `output_gate_type or hidden_act`; converter no longer silently creates an invalid GGUF for an unsupported nonsigmoid gate; unit tests passed; created `4b15fba8a` | Move to stage 9/final matrix |
| 2026-09-11 | Stage 9: Q8 ubatch | `[x]` | Q8+self-contained MTP at `ctx=524288`, `parallel=2`, `fit-target=3072` works with `ubatch=512`; long-prompt timing, acceptance, and post-decode VRAM recorded in the current configuration | Completed by final runtime matrix |
| 2026-09-11 | Stage 9: final runtime | `[x]` | Exact preset loaded in 64.39 s; 42.8k long prompt reached 689.49 cold / 837.58 warm tok/s; concurrent 2x1024 and single 4096 decode passed without OOM; VRAM and acceptance recorded | Review the updated plan; after commit, move to stage 10 |
| 2026-09-11 | Stage 9: Q8 KV autofit fix | `[x]` | Fixed unsigned underflow in MoE partition and stale self-contained MTP placement; Q8 KV fit 16.75-16.93 s, startup 49.33-49.57 s, 42.9k prefill and parallel=2 passed without OOM; CTest 6/6; created target commit `c5f079ab3` | Move to the final stage 10 matrix |
| 2026-09-11 | Stage 10: clean backend gate | `[x]` | New CUDA build completed; full CTest reached 63/64 because of a 25-minute backend-target timeout; found and fixed a false NaN from uninitialized KPOOL sentinels; KPOOL, TOP_K/indexer, and QSA pass on 6 GPUs; created `3bb01a85f` | Run final Q4/Q8 runtime matrix, MTP off/on |
| 2026-09-11 | Stage 10: Q4 short + multimodal MTP | `[x]` | Q4 text off/on deterministic and correct; vision off correct; reproduced Qwen+MTP HTTP 500 caused by GLM-only media catch-up; architecture-gated target-only fallback passed two vision->MTP text cycles and CTest 4/4; created `367bea7f1` | Q4 + shared MTP Q8 at 131k+ context |
| 2026-09-11 | Stage 10: Q4 structured 148k | `[x]` | Structured text of `148159` tokens retrieved all three markers exactly, acceptance 0.82796; repeat reused 148155 tokens and matched byte-for-byte; long vision of `148458` tokens in target-only mode read keys and headline exactly; post-vision MTP acceptance 0.90476; no OOM/HTTP 500; created plan commit `3e50c1fcd` | Q8 short text/vision off/on |
| 2026-09-11 | Stage 10: Q8 short text/vision | `[x]` | Text off/on correctly returned `472` and repeated byte-for-byte; MTP acceptance 0.90476, decode speedup 1.73-1.82x; vision off/on correctly read headline and Apollo 11; post-vision MTP acceptance 0.88889; health and VRAM clean; created plan commit `daebbe525`; later clarified that explicit `-ngl all` disabled the fitter | Q8 + self-contained MTP at 131k+ context with real autofit |
| 2026-09-11 | Stage 10: Q8 structured 148k | `[x]` | Real autofit without `-ngl/-ngld` loaded in 46.45-49.20 s; text retrieved all three markers exactly, acceptance 0.90260, prefill 604.75-609.04; repeat reused 148155 tokens and matched byte-for-byte; long vision read markers/headline exactly; post-vision MTP acceptance 0.88889; no OOM/HTTP 500; created plan commit `4f1ccff75` | Final parallel=2 matrix |
| 2026-09-11 | Stage 10: Q4 parallel=2 | `[x]` | ctx=524288 target-only and shared MTP passed isolated concurrent text, 2x1024 stress, and concurrent vision+text; media fallback applies only to its own slot, while the neighbor and both post-vision slots use MTP; at least 1776 MiB free; health OK; created plan commit `693ccdcbb` | Q8 parallel=2 off/on |
| 2026-09-11 | Stage 10: Q8 parallel=2 | `[x]` | Real target-only/self-contained MTP autofit passed; concurrent text isolated, MTP acceptance 0.90598-0.94839; concurrent vision+text confirmed per-slot fallback, both post-vision slots returned to MTP; at least 3098 MiB free; health OK; created plan commit `4cae0590f` | Check prompt-cache save/restore and rollback switches |
| 2026-09-11 | Stage 10: prompt cache and rollback | `[x]` | File-based target state saves and restores without error, but MTP draft/spec are absent and the first prompt is fully reprocessed; format change rejected. Current rollback switches checked against code; stale TOP_K name corrected | Review final plan update |
| 2026-09-11 | Additional rollback #28019 | `[x]` | Fixed full/partial reference test bug; localized real mismatch to the unwritten input-state GDN plane for a short batch; fix `26abcbf7e` passed real Q4 with `max diff 0`, synthetic Qwen4Exp, and recurrent CTest 4/4 | Update plan in a separate commit after review |

## Do Not Include Without a Separate Decision

- PR #28623: experimental and contains temporary/debug changes;
- closed alternative QSA PRs instead of selected #28213;
- unrelated changes from upstream master;
- automatic push or PR creation;
- PLE/image fix from the erroneous report finding.
