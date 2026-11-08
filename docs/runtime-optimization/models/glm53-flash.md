# Model passport: GLM-5.3-Flash UD-Q5_K_XL

Date: 2026-11-06. Runtime branch: `my_glm53_flash_1`. Parent before the combined GLM/MMQ commit: `19ca43137`; exact measured source/library identities are retained with the [experiment](../experiments/glm53-mmq-20261106/REPORT.md).

| Field | Value / evidence |
| --- | --- |
| Weights | `/home/sergei/.models/unsloth/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-Q5_K_XL-00001-of-00006.gguf`; unchanged six-shard model |
| Architecture | Legacy GGUF name `glm5next`, canonical runtime name `glm5-next`; 46 blocks including one MTP tail; hidden 4096; top 8 of 288 experts; expert FFN 2048; one shared expert |
| Attention | Hybrid KDA/mHC and sparse MLA; 64 heads, indexer 32 x 128, top-k 2048 with k-pool 4 |
| Precision observed in trace | Q5_K routed up/gate, Q6_K matmuls, Q8_0 dense matmuls; the GGUF is mixed precision |
| KV / workload | F16 K/V, unified context 557056; 50000 frozen text tokens, full state clear before each repetition |
| Backend | Five 48 GiB RTX 4090/4090 D and one 24 GiB RTX 3090; CUDA 13, SM86/89; hardware UUID/PCI identity in the local archive |
| Placement | All experts on GPU; 45 evaluated target layers distributed 11/8/8/8/8/2; matching 46-layer device maps before/after |
| Parameters | Batch 2048, ubatch 1024, threads 20, parallel 1, FA on, load mode none, fit target 512 MiB |
| Consumers | Plain text prefill, final-row logits; no drafter; no MoE expert cache |
| Result | Native median PP 836.820 -> 988.683 tokens/s (+18.1476%), three repeats per variant; sampled free VRAM >=1370 MiB |

## Applicability and validation

The shared CUDA MMQ path is active for this model, including fused up/gate/GLU. The pre-existing Q3_K tile setting is retained for its separately tuned shapes. MMQ dispatch and fusion remain conditional on weight types, geometry, operation semantics and device support.

Existing CPU-reference matrix/fusion tests passed on Ada and Ampere; the new fusion cases and GLM/Qwen4Exp architecture round trips passed on all six GPUs. Production binaries were rebuilt and checked. Final real-model logits are finite and repeatable within each runtime. Reduction/layout changes produce numerical differences across versions; the complete figures and test limits are in the report.

The measured workload uses repository text, not the user's original prompt. Full perplexity, other real model families, Blackwell/HIP, multimodal requests, drafter acceptance and throughput at longer prompt sizes are separate validation tasks. The ordinary GLM state/rollback/feature-export regressions are covered by the existing architecture tests.


## DFlash2 GPU feature transfer (2026-11-08)

Enabled by default for eligible GLM+DFlash2; `LLAMA_DFLASH_GPU_FEATURES=0` disables it and `1` explicitly enables it, with the Anbeeld `GLM-5.3-Flash-DFlash2-Q5_K_M.gguf` drafter and maximum4 draft tokens. Default ON was requested by the user after the explicit-ON experiment. Producer-local CUDA tap buffers and GPU packing replace host feature materialization; auto-fit counts their storage. Unsupported targets/layouts keep the host path; GLM input ownership remains synchronous.

The [separate DFlash2 experiment](../experiments/glm53-dflash-gpu-features-20261108/REPORT.md) uses ctx278528, B2048/U1024, unified F16 KV, P1 and the same saved50K documentation IDs +512 outputs, with three repeats per arm. Both arms use DFlash: PP904.157->953.046 (+5.407%), whole request63.635->60.771s; TG62.053->62.136 is within variation. Outputs, draft acceptance and actual tensor placement match. Synthetic slice/host/reset checks and the maintained GLM CPU/six-CUDA tests pass. About320MiB extra feature storage, minimum sampled free VRAM784MiB. This result does not update the historical plain/MMQ numbers above or compare Maya/MTP; prompt-suffix preparation, media and broader numerical-quality evaluation remain separate.


## Experimental protected prefill inputs (2026-11-08)

Default OFF. `LLAMA_GLM5NEXT_PIPELINE_PREFILL=1` permits protected scheduler input slots for eligible plain GLM contexts. The general pipeline still requires multi-CUDA layer placement and fully GPU-resident experts; `LLAMA_PIPELINE_GPU_OVERRIDES=1` was fixed in both measured arms. Abort/evaluation callbacks, embedding-output mode, LoRA, multiple sequences and layer/NextN consumers exclude this path. DFlash and MTP retain their previous behavior.

The [separate plain experiment](../experiments/glm53-prefill-pipeline-20261108/REPORT.md) fixes top-k2048, ctx278528, B8192/U1024, F16 unified KV, P1 and50K IDs +512 outputs. Median PP1019.222->2170.751, TG39.212->38.790 (a1.075% regression), whole request62.124->36.237s. Output tokens and actual weight placement match; state/lifecycle fixtures pass on CPU, two and six GPUs. Sampled minimum free VRAM824MiB. This does not update prior DFlash rates or establish media/MTP/DFlash/long-populated-context support. Committed at user request; no default promotion.
