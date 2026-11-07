# GLM-5.3-Flash DFlash2: GPU feature transfer

Date: 2026-11-08. Status: validated guarded default ON, promoted at the user's request after this experiment. The installed `my_build1` was not replaced.

Persistent local archive: `/home/sergei/_my_old/_my_glm53_flash/glm53-dflash-gpu-features-20261108/`. Raw artifact names below refer to this archive.

## Change

The target retains each tapped HC mean on its producer GPU. A collector on the drafter encoder's device packs these planes and passes a GPU tensor to the draft encoder. This reuses the existing DSpark machinery, extends its eligibility to GLM5-Next plus DFlash2, and accounts for the new buffers in auto-fit. The GLM path uses synchronous input ownership; the DSpark asynchronous-input switch does not enable asynchronous inputs for GLM. Unsupported configurations keep host features.

Eligible GLM+DFlash2 uses GPU features by default; `LLAMA_DFLASH_GPU_FEATURES=0` disables the path and `1` explicitly enables it. The native measurements below used explicit ON before promotion; promotion changes the unset-variable selection, not that path's arithmetic. Tested limits: one sequence, layer split, batch at most 8192, ubatch at most 1024, contiguous F32 feature producers on CUDA. Validation is specific to the real GLM UD-Q5_K_XL target and Anbeeld DFlash2 Q5_K_M. CPU, HIP, other target architectures, MTP and multimodal requests are not promoted by this result. Existing DSpark eligibility/defaults are preserved.

This does not implement prompt-suffix preparation. At the user's batch 2048 and drafter window 2048, discarding rows separately within each batch would save nothing. Deferring old-prefix draft injection across batches requires an explicit prompt lifecycle and checkpoint contract. It remains a separate change.

## Native result

Both arms use DFlash2 with four drafts. There is no fresh plain/MTP/Maya comparison in this series.

| Metric | Host features | GPU features |
| --- | ---: | ---: |
| Native PP median, tokens/s | 904.157 | 953.046 |
| PP range, tokens/s | 895.585-905.837 | 945.043-954.430 |
| Native TG median, tokens/s | 62.053 | 62.136 |
| TG range, tokens/s | 61.568-62.304 | 61.780-62.309 |
| Prompt median, seconds | 55.300 | 52.463 |
| Generation median, seconds | 8.235 | 8.224 |
| Full request median, seconds | 63.635 | 60.771 |
| Accepted/generated drafts, each repeat | 390/483 | 390/483 |
| Minimum sampled free VRAM, MiB | 818 | 784 |

PP improves 5.407%; complete request time falls 4.501%. TG +0.133% is within the observed variation and is not a demonstrated TG improvement. Additional declared feature buffers are about 320 MiB at B2048/U1024 with five 4096-wide taps; they are distributed across producer devices and the collector. Sampling every three seconds does not establish instantaneous memory peaks.

## Correctness and activation gates

- The real target and draft tensor names, buffer types/devices, element types, byte sizes and layer-device maps match between arms. `native-fit/analysis.json` records the comparison. Auto-fit does not confound this pair through a different weight placement.
- All six measured 512-output token arrays match, including the prior frozen continuation hash `2ee5a3f5c75ad68ef7335576132873b2eafd7064ee16b35acdc0c415a40bb38a` (SHA256 of JSON-serialized IDs). All acceptance counts also match. No prompt tokens are reused.
- The real ON log confirms `GPU feature bridge = enabled`; there is no host-packing fallback warning.
- A small native fixture compares packed slices byte for byte against the original host features, including offsets, invalid/duplicate taps, invalid ranges, host materialization and disable/reset. It passes with one visible CUDA device and a split over six visible CUDA devices.
- The maintained GLM architecture test adds equivalent packing/lifecycle checks on each CUDA device. `test-llama-archs --arch glm5-next` exits zero: CPU reference/state/roundtrip and six CUDA devices pass; unsupported Meta tensor split retains its skip.
- No precision, weights, attention selection, sampling or acceptance semantics were changed. Full perplexity, other corpora and media were not evaluated.

## Frozen protocol and identities

Checkout base: `1b986a317`, branch `my_glm53_flash_1`. Runtime snapshot: local `runtime/`, copied from the user's installed `my_build1` and replacing only `libllama` and `libllama-common` with the candidate build. The same snapshot and shared-library hashes are used in both arms, including the original installed CUDA library. Exact hashes are in `native-fit/identity.json`. Runtime patch and test patch identities are retained separately.

Original six-shard GLM-5.3-Flash UD-Q5_K_XL; Anbeeld GLM-5.3-Flash-DFlash2 Q5_K_M; unchanged model files. Five 48 GiB Ada GPUs and one 24 GiB RTX 3090, Threadripper PRO 3945WX; CUDA PCI bus order. Model paths and expanded commands are in each arm's `launch.json`.

Context 278528; B2048/U1024; threads 20; P1; unified F16 KV; FA on; normal auto-fit with 768 MiB target; no CPU MoE override; load mode none; CPU vision projector. Drafter maximum 4, minimum 0, p-min 0, all draft layers offloaded. Temperature .95, top-p 1, top-k 40, min-p .05, seed 1234. Saved 50000 repository-documentation token IDs, 512 outputs with EOS ignored. These are raw completion IDs, not the user's original chat prompt. Server prompt RAM cache disabled and each request uses `cache_prompt=false`.

Each fresh process runs an excluded 2048+64 warm-up followed by three clean native 50000+512 requests. Load time is excluded. No profiler, callback, compilation or GPU correctness test runs during measured requests. A startup-only interposer records model placement and does not hook decode or set a graph callback. There is one arm order and one corpus; no confidence interval or corpus-wide gain is inferred.

The monitor stops its own server below 512 MiB free VRAM, below 4 GiB available RAM or on a foreign GPU owner. Successful auto-fit runs exit zero, no guard fires, and both test servers are stopped afterward.

## Resource-gate attempts preserved

`native/`: the first manual split used the wrong virtual offload count and requested more VRAM than CUDA0 had. It failed before inference. GLM has 46 declared blocks including the MTP tail, plus the output offload position.

`native-fixed/`: the corrected whole-layer split loaded, but initialization/warm-up left only 306 MiB on CUDA4. The watcher stopped the server. No native speed result is claimed for either manual attempt; neither was a feature arithmetic failure. The completed experiment uses `native-fit/` and verifies actual tensor placement instead of assuming a manual split fits.

## Reproduction and artifacts

The archived opt-in snapshot predates default promotion and needs its own library path because the executable contains an absolute build RUNPATH. `run-server.sh` supplies that path and enables the opt-in switch; pass the usual server arguments. `LLAMA_DFLASH_GPU_FEATURES=0 run-server.sh ...` selects host features. The rebuilt production server in `build-aliceai/bin/llama-server` enables eligible GLM+DFlash2 GPU features without the flag.

`run-ab.py`, `analyze.py`, `check-features.cpp`, build logs, correctness logs, runtime snapshot, full streamed responses, timing JSONs, memory samples and placement JSONs are retained here. Initial copy/slice fixture: `check-features.log`; maintained architecture gates: `archs-glm.log` and `archs-glm-cross-ubatch.log`; successful comparison: `native-fit/analysis.json`. Runtime source patch is `candidate.patch`; the final patch including the maintained test is `candidate-with-test.patch`.


## Default promotion checks

The final default-ON server and architecture test were rebuilt. With `LLAMA_DFLASH_GPU_FEATURES` unset, the GLM architecture suite passes on CPU and all six CUDA devices, including packed slices crossing an ubatch boundary, host materialization and disable/reset. The log is `default-on-archs.log` in the local archive. These checks do not introduce a new throughput series; the figures above remain the frozen explicit-ON experiment.
