# Qwen3.8-Flash-Next upstream port, 2026-11-06

Status: implemented and measured on `my_glm53_flash_1`. Measurements and source/binary identities were captured from the working tree before this commit. Production `llama-server`, `llama-cli`, `test-llama-archs`, and `test-backend-ops` were rebuilt in `build-aliceai`. No merge or push was made.

Base: `a4968c98445d0bb5a47d0575aa9be898df5135a2`. Upstream source: local original at `e14a779719652200cec7ad58dfa4820edc347182`. The original repository and model files were not modified.

## Result

Direct native execution without Nsight, one warmup and three measured repetitions. Each repetition clears memory and recomputes the same saved 21,000 token IDs; decode is 64 fixed teacher-forced tokens. Units are tokens/s.

| Runtime | PP median [min, max] | TG median [min, max] |
| --- | --- | --- |
| Before port | 1911.846 [1909.294, 1913.182] | 53.030 [52.972, 53.242] |
| Final port | 4499.916 [4490.426, 4501.106] | 63.751 [63.360, 63.891] |
| Fresh upstream control | 4819.886 [4809.248, 4837.805] | 73.122 [73.024, 73.260] |

The port improves PP by 135.3702% (2.3537x) and TG by 20.2169%. It remains 6.6385% below the fresh upstream PP control and 12.8156% below its TG control. This is an implemented port with a remaining performance gap, not a claim of parity.

The text is repository documentation, not the user's original prompt. The user's prompt length of 21,000 is reproduced, but absolute server figures 1483/45 and 3263/66 are not expected to match this harness. Chat formatting, sampling, mmproj, and draft execution are excluded from the native test.

## Frozen setup and identity

Model: `/home/sergei/.models/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf`. GGUF architecture `qwen4exp`, six shards, 48 trunk blocks, 36 GDN and 12 QSA blocks; 10 selected experts of 512; compression ratio 4 and indexer budget 4096. The file already contains budget 4096. Model weights were not rewritten.

Context 262144, batch 2048, ubatch 512, parallel 1, unified F16 K/V, FA enabled, compute/batch threads 20, mmap, lazy AUTO, cache-ram 32768 MiB and fit target 512 MiB. All layers and the output head are on GPU; placement remains 9/9/9/9/9/3 on CUDA0-5. Pipeline parallelism is enabled, with four scheduler copies and no tensor overrides. The CPU PLE table remains lazy-mapped.

Hardware: five modified 48 GiB RTX 4090/4090D and one 24 GiB RTX 3090, Threadripper PRO 3945WX. Source, executable, library, build cache, input, model-header and hardware fingerprints are retained in the local artifacts. The before-port baseline and original model inventory remain in `/home/sergei/_my_old/_my_glm53_flash/qwen38-audit-20261106`.

Runners check for existing compute processes before loading, sample per-GPU memory, and stop only their own process group if free memory drops below 512 MiB. All final native, profile and state runs exited successfully. CPU builds and checks finished before timed native processing; the fresh reference waited behind a gate while CPU state checks ran. No other GPU job was used for a benchmark.

## Implementation

### QSA and pooled-key lifecycle

The model now stores raw and pooled keys in two-column indexer rows, recomputes only newly completed or invalidated pools, and uses upstream Top-K, fused selection masks and sparse-attention selection bounds. CPU cache inputs are shared by the QSA layers. Pools follow token cell order, including distinct M-RoPE cells with the same time position. The GLM three-column layout and position-based grouping remain the default for GLM.

The shared index-cache code retains our partial-state gates, restore failure cleanup, MTP APIs and GLM support. The old QSA mask/gather builder was removed. The old `QWEN4EXP_FUSED_LID` and `QWEN4EXP_QSA_GATHER` switches are superseded by the new QSA graph.

Our MTP metadata inference, optional shared embedding/LM head and `QWEN4EXP_MTP_QSA` dense fallback remain. `QWEN4EXP_MTP_TOPK_SHARE` reuses a selection only in a context configured for one sequence, with one token and stream. Multiple sequences use fresh selection because appends can shift the global pool indices. Even a reused selection updates raw keys and pooled cache state. Existing MTP checks were adjusted to verify those writes rather than requiring their omission.

Top-K uses the existing upstream radix implementation for eligible smaller multi-row selections, including the actual Qwen k=1024 workload. The previously tuned k=512/2048 paths remain. Main MMQ kernels were already ported in the base commit and were not replaced again.

### GDN MMA and mHC

The upstream GDN MMA source and required shared-memory swizzle helpers were added. Its final-state-only path is restricted to K=1. All requests for multiple recurrent snapshots retain the existing AR implementation. This is stricter than the copied upstream eligibility expression and preserves our rollback contract. Existing KDA kernels and unsupported-device fallbacks remain.

Qwen mHC pre/post fusion is enabled for prefill-sized batches. CUDA also recognizes the scale/sigmoid/scale chain feeding HC post, with the upstream edge, consumer and memory-range checks. The actual model has separate projection tensors; optional upstream merged projection formats were not required or introduced by this port.

### Root cause of repeated graph planning

Kernel changes alone reached only about 2042 PP / 61.6 TG. A separate allocator diagnostic identified the principal remaining PP bottleneck: the initial worst-case plan had 7082 nodes, while a no-output chunk had 7081. The different topology discarded the full-context plan. Each growing pool tensor then exceeded the just-created small plan, causing another reserve and global backend wait.

A first change retaining empty reductions fixed the cold first prefill but not later requests: single-row decode had only 7034 nodes and discarded the reserve again. That incomplete variant is preserved in the evidence.

The final fix retains conservative allocation dependencies for empty and single-row MoE reductions during graph optimization. Execution still uses the previous kernel-selection policy. No synchronization or lifetime check was removed. Prefill and decode now keep the same 7082-node allocation topology.

The final two-cycle diagnostic records 86 allocation calls, zero reserve/replan operations, and zero time in reserve-related synchronization. The prior audit measured 42 replans and 3.898793 seconds of reserve-related waits for one PP. Other necessary input/output and stream synchronization still exists.

## Verification

| Gate | Result and scope |
| --- | --- |
| Build | Release CUDA SM86/89 server, CLI and existing tests pass |
| Architecture checks | Qwen, GLM and DeepSeek41 pass on all six GPUs and CPU references; Qwen also passes after the final MTP eligibility guard |
| MTP | Synthetic Qwen MTP, shared head, two-sequence fresh-selection fallback and rejection/replay checks pass |
| GDN | CPU-reference checks on Ada/Ampere pass, including Hk16/Hv48/D128/T512, a permuted T65/B2 tail and K=4 snapshot fallback |
| mHC and masks | Existing and added gated HC post cases, actual 2560x512 geometry, HC pre and Top-K mask cases pass on Ada/Ampere |
| Top-K | Actual 5376x512/k1024, tied scores and single-row checks pass on Ada/Ampere |
| State snapshots | Full, per-sequence, partial plus suffix trim, sequence copy and recovery after a truncated full snapshot pass on a synthetic Qwen GGUF, CPU/Ada/Ampere |
| M-RoPE ordering | An eight-patch embedding input with repeated time positions matches a 4+4 split on the synthetic model; CPU is exact, Ada/Ampere NMSE is below 1e-6 |
| Full-model screen | Finite full-vocabulary logits for all native runs; short teacher-forced NLL and upstream repeat controls retained |
| Activation | Final Nsight trace confirms GDN MMA and gated mHC kernels, with no allocator replans across PP/TG/PP/TG |
| Memory | Final sampled per-device memory and free-memory minima retained; no memory-floor stop |

State tests use the existing architecture-test GGUF generator. No new repository test file was introduced. A temporary M-RoPE helper initially under-allocated its position array; it was fixed to supply 4*N positions and rerun. The failed helper and log are preserved as a harness error, not a runtime failure.

Full-model logits are not bit-identical across implementations or across upstream repeats. For final run 1 versus the fresh reference, next-token top-1 matches at 62/64 teacher-forced rows; raw max/RMS errors and repeat controls are retained. NLL over 64 tokens is a screen, not a full perplexity or task-quality evaluation. No universal numerical acceptance threshold is inferred from another model's history.

Not tested: actual image files through mmproj, real-model draft acceptance/TG, full task/PPL evaluation, Blackwell/HIP/MUSA execution, larger real contexts or other quantizations. Synthetic M-RoPE and MTP gates do not replace those tests.

## Experiment history and remaining gap

| Variant | Warm PP median | Warm TG median | Decision |
| --- | ---: | ---: | --- |
| QSA only | 2077.934 | 61.426 | Retained as the first measured stage |
| QSA + GDN + mHC, old allocator policy | 2042.039 | 61.605 | Correct kernels, unresolved PP bottleneck |
| Shape-cache OFF on that stage | 2050.514 | 65.428 | Negligible PP gain; not promoted |
| Empty-only allocation fix | 2044.147 | 62.272 | First cold PP improved, repeated requests still failed the speed goal |
| Final conservative allocation dependencies | 4499.916 | 63.751 | Implemented measured candidate |

The shape cache remains at its prior default. The single-token weighted-reduction execution policy and CUDA graph chunking still differ from upstream; their exact contribution to the remaining gap was not isolated here. No unmeasured selector was enabled to claim parity.

The local archive is `/home/sergei/_my_old/_my_glm53_flash/qwen38-port-20261106`. It includes raw logs and timings, frozen tokens, logits, profiles, helper sources, failed variants, source patches, final source/binary identities and SHA256 manifests. `qsa-stage` and `kernels-stage` preserve separate source and binary snapshots. The model weights are referenced in place, not copied. Production source and defaults are represented by the final working-tree patch, not by the base commit alone.
