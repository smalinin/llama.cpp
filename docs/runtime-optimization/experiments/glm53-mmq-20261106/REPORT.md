# GLM-5.3-Flash: upstream CUDA MMQ port

Date: 2026-11-06. Status: shipped default, guarded by the normal CUDA dispatch. The baseline was `my_glm53_flash_1`, parent `19ca43137`, with the GLM graph/TOP_K_MASK/KDA changes already applied. The candidate adds the MMQ port; the final commit also includes those preceding GLM changes.

## Change and applicability

Ported the MoE tile layout and gate/up/GLU MMQ kernels from local upstream `e14a77971`, principally `c49edf096`, `f5222f25c`, `5ce06fc71` and `69ee2ecc0`. MMF uses the corresponding tile descriptors because it shares the expert-index helper. The existing Q3_K tile control, architecture configurations, activation precision policy and decode projection fan remain supported. No new public precision API is required.

This is a shared CUDA path for compatible dense FFN and MoE graphs, rather than an architecture-specific optimization. Fused weights must have matching type, shape and strides, supported GLU semantics and suitable tile geometry. Unsupported cases use the existing dispatch. Fused types are Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS and NVFP4. NVIDIA fusion requires Turing MMA or newer. Blackwell, HIP and other backend hardware were not tested here.

The allocator keeps the fused inputs alive through the GLU, including copied weights. Fusion does not split the K reduction. The linearized MoE layout changes tiling/reduction scheduling; cross-version bitwise equality is not required. Existing operator error limits were retained without relaxation.

## Frozen protocol

- Same six-shard `GLM-5.3-Flash-UD-Q5_K_XL` model in `/home/sergei/.models/unsloth/GLM-5.3-Flash-GGUF/`; no weight conversion.
- Same saved 50,000 token IDs, generated from repository documentation, rather than the user's original prompt. Token SHA256 is in the result/identity manifests.
- Context 557056; batch 2048; ubatch 1024; threads 20; parallel 1; unified KV; Flash Attention enabled; F16 K/V; load mode none; fit target 512 MiB; no drafter or MoE expert cache.
- Auto-fit selected identical placement for all 46 model layers. The 45 evaluated target layers are distributed 11/8/8/8/8/2; the remaining layer is the unused MTP tail.
- Five 48 GiB RTX 4090/4090 D devices and one 24 GiB RTX 3090. CUDA 13.0; Release; architectures 86-real/89-real; FA and CUDA graphs enabled; CPU and CUDA build options matched between the measured variants.
- One profiled pass, then three native repeated passes per variant. Model states were cleared before every pass; only the final token requested logits. Timing excludes model loading and includes final synchronization. No compiler or correctness test ran during timed prefill.
- GPU memory/temperature/clocks were sampled once per second. The runner stops its own process group if free VRAM falls below 512 MiB. Both runs exited zero and released their GPU allocations.

## Correctness and activation

| Check | Result |
| --- | --- |
| New upstream FFN/MoE fusion cases | Passed against CPU on all six GPUs |
| Full existing MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION suites | Passed on CUDA0/Ada and CUDA5/Ampere; unsupported cases retain normal skips |
| GLM and Qwen4Exp architecture/state round trips | Passed on CPU and all six GPUs |
| Production rebuild checks | GLM, Qwen4Exp and new fusion cases on Ada/Ampere passed |
| Real 50K-token outputs | Finite; four passes within each runtime produced identical final logits |
| Candidate vs baseline final logits | NMSE 0.00692889; max absolute difference 0.982663; same argmax; KL(reference,candidate) 1.74190e-7 nats |
| Full perplexity, multiple prompt corpora, media and drafter acceptance | Not measured in this experiment |
| Fusion in real-model trace | 0 baseline launches; 3034 candidate fused MMQ launches |

The final-token comparison is a numerical diagnostic and does not replace a perplexity evaluation. Synthetic operator checks use the unchanged existing CPU-reference error limits. Architecture tests exercise state, rollback and feature-export behavior in the existing GLM test suite.

## Native speed and memory

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Native PP, median of 3, tokens/s | 836.820 | 988.683 |
| Native PP range, tokens/s | 835.565-839.365 | 985.421-989.727 |
| Native median time, seconds | 59.749995 | 50.572351 |
| Profiled PP, tokens/s | 831.753 | 979.580 |
| Minimum sampled free VRAM, any GPU, MiB | 1806 | 1370 |
| Sum of MMQ GPU kernel durations, seconds | 29.448971 | 21.288799 |

Native throughput gain is **18.1476%** on this frozen GLM workload. GPU kernel sums are diagnostic durations across devices, not wall time. The memory minima cover the whole observed process lifetime; the candidate observation also includes small production correctness tests while the loaded model waited at its gate. They are conservative sampled minima, not a proof of instantaneous peaks.

Both traces preserve the existing attention/indexer implementations. These results do not assert an equal speedup for every compatible model or for token generation.

## Evidence and reproduction

Persistent local archive: `/home/sergei/_my_old/_my_glm53_flash/glm53-mmq-20261106/`.

The archive contains `glm53-mmq-results.json`, source/library identity manifests, the exact MMQ patch, frozen token IDs, benchmark source/compiler helper, expanded command arrays, memory CSVs, correctness logs, production build logs, final logits, Nsight reports and SQLite exports. The baseline identity records the dirty source patch and library hashes before the production libraries were rebuilt. No model shards or compiled shared libraries are archived there.

Run `glm53-mmq-analyze.py` with the corresponding raw files restored to its `/tmp` paths to regenerate the metrics. The original llama.cpp checkout was not changed. Source review and `git diff --check` passed. See the [model passport](../../models/glm53-flash.md) for the model-specific scope.
