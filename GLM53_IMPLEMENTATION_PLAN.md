# GLM-5.3 Implementation Plan

## Scope

This plan covers the next two development tracks in
`/home/sergei/Github/llama.cpp` on branch `my_build_glm53_flash`:

1. Adapt the useful GLM-DSA/MTP improvements to GLM-5.3-Flash (`glm5_next`).
2. Port the GLM-5.2 `glm-dsa` fixes to standard GLM-5.3 (`glm_moe_dsa`).

The target branch starts this work at `cc6dea3b2`. The existing GLM5NEXT work
was originally based on Unsloth's
[`glm5next/upstream`](https://github.com/unslothai/llama.cpp/tree/glm5next/upstream)
branch. The reference GLM-DSA fixes are available in
`origin/my_build_glm52`, ending at `0b5c2f91a`.

## Architecture boundary

| Model | llama.cpp architecture | Indexer layout | Relevant implementation |
| --- | --- | --- | --- |
| GLM-5.3-Flash | `LLM_ARCH_GLM5NEXT` / `glm5_next` | Full indexer in every sparse layer | K-pool cache and custom indexed FlashAttention |
| GLM-5.3 | `LLM_ARCH_GLM_DSA` / `glm_moe_dsa` | Full and shared indexer groups | DSA cache and generic sparse FlashAttention |

The two paths must remain separate. In particular, shared-indexer mask reuse is
useful for standard GLM-5.3 but does not apply to GLM-5.3-Flash.

## Working rules

- Adapt changes to the current APIs; do not blindly cherry-pick old commits.
- Preserve CPU and unsupported-backend fallbacks.
- Add an A/B switch for a new fast path whenever a safe fallback is practical.
- Each implementation stage gets its own code commit.
- After each code commit, update `GLM5NEXT_LOCAL_CHANGES_EN.md` in a separate
  documentation commit, then stop for review before starting the next stage.
- Do not include merge commits from `origin/my_build_glm52`.
- Record correctness and performance results using the same prompts, sampling
  seed, context, cache types, slot count, and draft parameters.

## Phase 0: Freeze the baseline

This phase changes no production code.

- Build the current CUDA configuration and record the build flags and GPU model.
- Run the existing backend, architecture, K-pool, speculative, and server tests.
- Record GLM-5.3-Flash results with MTP disabled and enabled at short context,
  50K, 80K, and 100K context.
- Record prefill speed, generation speed, draft acceptance, mean accepted length,
  VRAM use, and CUDA graph reuse for `n_slots = 1` and `n_slots = 2`.
- Keep the raw baseline logs outside the repository unless a compact benchmark
  summary is worth committing.

Exit criterion: a reproducible baseline exists and the current test suite passes.

## Phase 1: Useful adaptations for GLM-5.3-Flash

### F1. Add regression coverage before changing the fast path

Reference ideas:

- `8b38bc580` - sequential GLM MTP drafting coverage.
- `ab345947d` - multi-stream sparse FlashAttention coverage.

Adapt the tests to `glm5_next` and its custom `fattn-indexed` operation:

- Verify sequential MTP steps, reuse enable/disable transitions, accepted-token
  catch-up, rejection, rewind, and a fresh request after slot reuse.
- Exercise indexed attention with at least two streams/sequence IDs, including
  different valid KV lengths and invalid/padded indices.
- Cover both F16 and Q8_0 KV-cache paths where supported.

Commit: `tests: cover GLM5NEXT sequential MTP and multi-stream indexed attention`

Exit criterion: tests pass under CUDA compute-sanitizer where practical and do
not change model output.

### F2. Skip redundant indexer key and gate work during MTP reuse

Reference idea: `0fd68f6c6` (`glm-dsa: skip reused MTP indexer keys`).

The current GLM5NEXT graph reuses top-k after the first MTP iteration, but still
computes and writes the indexer key and compressor gate. Move the reuse decision
before this work when the cached selection is valid.

Requirements:

- Keep the incomplete tail and newly accepted positions correct.
- Mark modified pools dirty and rebuild them during target catch-up.
- Isolate cache state per stream/slot.
- Reset reuse on rejection, rewind, sequence reset, prompt reprocessing, and
  context shift.
- Retain `LLAMA_GLM5_MTP_TOPK_SHARE=0` as the A/B/correctness fallback.

Commit: `glm5next: skip redundant MTP indexer key updates`

Exit criterion: identical greedy output and acceptance decisions with reuse on
and off; lower MTP graph time without regressions for multiple slots.

### F3. Keep the complete MTP inner loop on the backend

Reference idea: `f09e46594` (`glm-dsa: keep MTP draft state on backend`).

GLM5NEXT already keeps hidden-state transfer on the GPU, but each MTP step still
synchronizes the draft context and feeds the sampled token back through the CPU.
Adapt the device-resident draft loop to the existing GLM5NEXT multi-slot hidden
buffers and adaptive draft limits.

Requirements:

- Keep sampled token/probability state and the next hidden-state input on the
  backend between MTP iterations.
- Read back only the compact final result required by speculative verification.
- Preserve `spec-draft-n-max`, `n_min`, `p_min`, and remaining-context limits.
- Support independent per-sequence ranges for `n_slots > 1`.
- Keep the current host-driven path as a runtime fallback.
- Do not enable the path for unsupported samplers, multiple NextN heads, or
  incompatible shared-memory/chain-head configurations.
- Preserve the multimodal GPU-direct embedding path.

Commit: `glm5next: keep MTP draft iterations on backend`

Exit criterion: no per-step host synchronization in the supported path; matching
tokens and acceptance versus the fallback; measurable decode improvement at
short and long context for one and multiple slots.

### F4. GLM5NEXT validation checkpoint

- Run the Phase 0 matrix again.
- Add multimodal prompt reuse and multimodal generation cases.
- Check cancellation, slot reuse, LCP rewind, context shift, and auto-fit/OOM
  boundaries.
- Compare MTP on/off and every new A/B switch.

Checkpoint status: completed; awaiting review. Validation exposed and fixed two
independent problems:

- `803b76c92` corrected backend-resident MTP state across context shifts. A
  7800-token prompt plus 512-token generation crossed an 8192-token context
  boundary with identical device/host-fallback output and acceptance.
- `b831f4647` corrected the auto-fit logical layer range when NextN tensors are
  intentionally skipped. Before the fix, MTP-disabled GLM5NEXT was reported as
  `46/47` layers offloaded, left layer 0 on the host, and disabled fused HC-pre.
  After the fix it reports `47/47`, keeps all target layers on GPU, and enables
  all three fused HC operations.

The single-slot F16 MTP matrix reached 565.38/56.46, 567.06/53.14, and
550.36/50.83 prompt/decode tokens per second at 50K, 80K, and 100K,
respectively. Q8_0 produced identical tokens but was slower on this hardware:
518.02/48.22, 487.56/52.06, and 465.96/49.82 tokens per second. Disabling CUDA
Graphs at 50K preserved output and reduced decode from 56.46 to 54.43 tokens per
second. With corrected target placement, the 50K MTP-off reference produced
identical tokens at 626.04 prompt and 32.71 decode tokens per second; MTP
therefore improved decode by 72.6% while adding 9.7% prefill overhead in this
test.

Two simultaneous 50K, 80K, and 100K slots produced identical target tokens to
each other and to their single-slot references, with no state/position errors
or OOM. The 80K and 100K makespans corresponded to aggregate request throughput
of 589.76 and 555.92 tokens per second. Per-slot decode timings are intentionally
not treated as isolated kernel results: the server scheduler can pause one
slot's generation while the other performs a large prefill. Loading two
106496-token slots (`n_ctx = 212992`) also exercised auto-fit beyond the target
192K production context without an allocation failure.

The normal production sampler (`temperature = 0.8`, `top_k = 40`,
`top_p = 0.95`, `min_p = 0.05`, fixed seed) produced byte-identical text and all
256 identical target tokens with GPU-direct drafting and the host fallback.
Both runs accepted `191/191` drafts; GPU-direct measured 76.36 versus 74.75
decode tokens per second at 8K context (+2.15%), while prefill was unchanged.
Cancellation, slot reuse, text LCP rewind, multimodal prompt reuse/generation,
and the context-shift boundary also completed successfully.

The CUDA Release build, 62 of 63 CTest entries, the complete CUDA0 focused
matrix (`548/548`), and CUDA5 HC/K-pool checks (`14/14`) passed. The exhaustive
multi-GPU `test-backend-ops` entry reached its 1500-second CTest timeout without
an observed correctness failure. These results satisfy the F4 validation exit
criterion; review is required before Phase 2 starts.

No optimization work starts in Phase 2 until this checkpoint is reviewed.

### F5. Fair long-prompt scheduling for multiple slots

Status: completed; awaiting review.

Commit `4431b580d` replaces first-slot prompt monopolization with round-robin
scheduling of complete configured prompt quanta. Keeping a full per-sequence
quantum is important for recurrent/hybrid models: the rejected intermediate
fair-share implementation split every 4096-token batch into 2048-token chunks
and reduced one slot's 100K MTP acceptance to `61/99`. The final scheduler
alternates full 4096-token chunks, while the backend MTP boundary allocation is
initialized to the zero state used by the host fallback.

The controlled 20K transition test retained `95/95` acceptance in both slots.
The final simultaneous 100K run used `n_ctx=212992`, `n_slots=2`,
`batch=4096`, `ubatch=2048`, F16 KV, and MTP `n_max=3`. Both prompts completed
together at 269.20 and 265.91 tokens/s; generation measured 31.98 and 30.61
tokens/s with `95/95` accepted drafts in each slot. Both responses were
byte-identical to the single-slot greedy reference. This removes the previous
37.50/0.73 tokens/s scheduler imbalance without changing model output.

`test-batch-alloc` and `test-llama-archs` passed after the implementation.

## Phase 2: Port the `glm-dsa` fixes for standard GLM-5.3

The official GLM-5.3 configuration uses `GlmMoeDsaForCausalLM` with
`model_type = glm_moe_dsa`, one NextN layer, `index_topk = 2048`, and both full
and shared indexer groups. The port therefore targets `src/models/glm-dsa.cpp`,
not `src/models/glm5next.cpp`.

### D0. Structural audit against the current branch

Status: completed; awaiting review. The table below was audited against
`7313bbb19` on `my_build_glm53_flash`. `Missing` means that the behavior is not
present for `LLM_ARCH_GLM_DSA`; it does not imply that an old patch can be
cherry-picked without adaptation.

| Reference hunk | Current-branch finding | Classification | Transfer decision |
| --- | --- | --- | --- |
| `a569338e3`, `common/fit.cpp`: include NextN in the logical layer range | Current fit uses `n_layer + n_layer_nextn`; `b831f4647` additionally documents and validates skipped-NextN placement | superseded | Keep the current implementation; validate GLM-DSA near OOM in D7 |
| `a569338e3`, `llama-model.cpp`: preserve `TENSOR_SKIP` in `create_tensor_qkv` | The same early skip path is present in the current loader | already present | No transfer |
| `ec699a26a`, server: provisionally fit target before measuring MTP | `common_fit_extra_model` now measures target and draft/MTP together and carries the fitted target placement into the extra context | superseded | No transfer |
| `7f39659e6` / `204e32d97`: conditionally/unconditionally alter the NextN layer count | These are intermediate revisions of the older fit scheme; copying either would bypass the current shared-extra-model accounting | superseded | No transfer; D7 is validation-only unless a reproducible gap remains |
| `b6fa6a70b`, GLM-DSA norm epsilon | `glm-dsa.cpp` loads RMS epsilon but still does not initialize the LayerNorm epsilon used by the indexer K norm | missing | Set the architecture value in D1 and cover CPU/CUDA logits |
| `b6fa6a70b`, indexer tensor flags | Trunk shared-indexer tensors are optional, but the NextN block's full indexer is also incorrectly optional | missing | Make full trunk and loaded MTP indexers required in D1; retain trunk-only/MTP-only loading |
| `b6fa6a70b`, optional DSA graph inputs | The indexer rotation input is already guarded; MLA/LID indices and masks are still written unconditionally even when the dense graph leaves an input unallocated | missing (partial) | Add buffer guards for the remaining optional inputs in D1 |
| `b6fa6a70b`, dense threshold and nullable `top_k` | GLM-DSA always evaluates the indexer and `build_attn(DSA)` assumes non-null `top_k`, even while all KV entries fit inside `indexer_top_k` | missing | Skip scoring and use ordinary dense MLA below the sparse crossover in D1 |
| `b6fa6a70b`, GLM-DSA MTP cache and graph | The MTP context still allocates a plain K-only cache and the MTP graph runs dense MLA without its full indexer | missing | Allocate a DSA cache and build the full indexed MTP block in D1 |
| `b6fa6a70b`, disable GLM-DSA cache shifting | Both DSA sub-caches currently report shift support for GLM-DSA | missing | Disable shifting for this architecture in D1 until both MLA and indexer state have a proven shift transform |
| `b6fa6a70b`, FlashAttention `src[5]` top-k path | Current ggml has the newer explicit-index contract, mask compaction, bounds checks, sequence-aware addressing, and sparse MMA dispatch | superseded | Reuse the current sparse FlashAttention APIs; do not copy the old CUDA templates or `src[5]` hint API |
| `b6fa6a70b`, multi-row CUDA top-k dispatch | The CUB `DeviceTopK` path still launches per row; the old segmented argsort helpers are absent | missing, performance-only | Keep out of D1 correctness work; profile after D6 before deciding on a separate generic CUDA commit |
| `b6fa6a70b`, CCCL `>= 3.2` preprocessor check | The current check still mishandles future major versions whose minor version is below 2 | missing, generic | Carry the small version check with the first top-k CUDA change, not the architecture patch |
| `b6fa6a70b`, GLM-DSA dense/MTP architecture tests | Current synthetic MTP and reuse coverage is enabled only for GLM5NEXT | missing | Add GLM-DSA dense, sparse, MTP, and missing-indexer cases in D1/D2 |
| `0be18c8ca`, optional top-k hint contract and OOB hardening | Explicit sparse indices are now a first-class FlashAttention input; current CUDA validates/ignores invalid entries without making the mask a hidden hint contract | superseded | No transfer |
| `ab345947d`, multi-stream sparse FlashAttention test | Current tests cover multi-stream GLM5NEXT indexed attention, but not the generic mask-compaction route used by GLM-DSA | missing (partial) | Add the GLM-DSA/generic sparse variant in D1 |
| `e9595867d`, `index_share_for_mtp_iteration` GGUF metadata | `indexer_types` conversion/loading already exists; only the standard GLM-DSA MTP-sharing key and hparam are absent | missing | Add converter, GGUF writer, saver, loader, and backward-compatible default in D2 |
| `e9595867d`, host MTP top-k capture/reuse state machine | GLM5NEXT exposes reuse through its memory object, but GLM-DSA has no capture/reuse state | missing | Adapt as the correctness fallback in D2; do not mix it with GLM5NEXT K-pool state |
| `f5b08c276`, require a full NextN indexer | The present MTP loader/graph accepts a missing or shared NextN indexer | missing | Reject incomplete/incompatible MTP heads and test the failure in D2 |
| `8b38bc580`, stable capture output and scheduler synchronization | GLM-DSA has no top-k capture output yet; the old lifetime/synchronization fix therefore is not present | missing | Incorporate the final safe host-capture form directly in D2, with sequential drafting coverage |
| `aa96a2059`, backend-resident per-sequence MTP top-k | Generic backend buffer allocation machinery exists for GLM5NEXT draft token/hidden state, but there is no GLM-DSA top-k buffer | missing (infrastructure reusable) | Add independent per-sequence GLM-DSA top-k storage in D3 using current buffer APIs |
| `0fd68f6c6`, persist sharing metadata in model saver | The standard GLM-DSA sharing field does not exist yet | missing | Include the saver hunk with D2 metadata support |
| `0fd68f6c6`, skip indexer K/score/write during reuse | The GLM-DSA MTP graph always executes the indexer path | missing | Implement only after D3 cache validity is backend-resident, in D4 |
| `f09e46594`, speculative device loop and persistent token/hidden/result buffers | F3 already added the generic loop, adaptive limits, multi-slot buffers, fallback, and public/internal API; it is intentionally gated to GLM5NEXT | already present for infrastructure; missing for GLM-DSA | Reuse the infrastructure and widen only validated architecture gates in D5 |
| `f09e46594`, GLM-DSA device-draft graph adapter | Only `glm5next.cpp` can currently read/write the persistent device-draft state | missing | Add an architecture-specific GLM-DSA adapter and tests in D5 |
| `0b5c2f91a`, prebuild and reuse the sparse mask across shared indexer layers | GLM-DSA reuses `top_k` but reconstructs the full sparse mask in every shared layer | missing | Extract current-API mask construction and reuse it per full/shared group in D6 |
| `0b5c2f91a`, old `build_attn` call-site churn for DeepSeek32 | Current attention signatures and sparse APIs have changed since the reference branch | not applicable | Do not reproduce mechanical old-API changes; keep unaffected architectures unchanged |
| `b028e9623`, CUDA radix selection for `k <= 512` | The kernel is absent on the NVIDIA CUB path (a separate HIP radix path exists), while standard GLM-5.3 requests `topk = 2048` | not applicable to the target configuration | Leave deferred; reconsider only with a measured generic top-k bottleneck and a design that covers `k = 2048` |

D1 therefore needs no old FlashAttention kernel transplant. Its production-code
scope is the GLM-DSA loader, graph/cache wiring, dense/sparse crossover, and
focused tests. D2 owns the missing standard metadata and the safe host fallback;
D3-D6 then remove the identified overheads in dependency order.

Exit criterion: met. Every local hunk in the GLM-DSA reference series is either
assigned to D1-D7, retained in deferred work, or explicitly superseded/not
applicable.

### D1. Port the GLM-DSA correctness foundation

Reference: `b6fa6a70b`.

Adapt only the portions still missing from the current branch:

- Correct dense-attention Q/K/V inputs.
- Correct sparse-versus-dense dispatch.
- Preserve sequence-aware KV mapping and bounds in sparse FlashAttention.
- Bring the GLM-DSA graph and cache interfaces to their final current-API form.
- Add focused architecture/backend tests for every transferred correction.

Commit: `glm-dsa: fix attention inputs and sparse dispatch for GLM-5.3`

Exit criterion: dense and sparse reference comparisons pass for single- and
multi-stream batches without changing unaffected architectures.

### D2. Add safe MTP top-k reuse and require a full NextN indexer

References:

- `e9595867d` - reuse top-k across MTP steps.
- `f5b08c276` - require a full MTP indexer.
- `8b38bc580` - sequential MTP regression coverage.

Implement the final corrected behavior directly:

- Preserve the already-supported `indexer_types` metadata and add the missing
  standard GLM-DSA `index_share_for_mtp_iteration` metadata.
- Allow MTP top-k reuse only when the NextN indexer is full and compatible with
  the target selection.
- Reset reuse at all sequence-state boundaries.
- Add sequential exactness tests before enabling the optimization by default.

Commit: `glm-dsa: safely reuse full-indexer top-k during MTP`

Exit criterion: MTP output and acceptance match reuse-disabled execution across
accept/reject/catch-up sequences.

### D3. Keep the GLM-DSA MTP top-k cache on the backend

Reference: `aa96a2059`.

- Store top-k state in backend buffers.
- Keep cache ranges independent per sequence/slot.
- Avoid full top-k CPU round trips.
- Preserve a host fallback for unsupported backends.

Commit: `glm-dsa: keep MTP top-k cache on backend`

Exit criterion: no host copy of the full top-k array on the supported CUDA path;
single- and multi-slot tests pass.

### D4. Skip reused GLM-DSA MTP indexer keys

Reference: `0fd68f6c6`.

- Skip indexer-key work only while the cached MTP selection remains valid.
- Recompute affected entries after acceptance, rejection, rewind, or cache
  mutation.
- Cover final partial groups and context-boundary behavior.

Commit: `glm-dsa: skip reused MTP indexer keys`

Exit criterion: identical results to the non-reuse path with reduced per-draft
graph work.

### D5. Keep GLM-DSA MTP draft state on the backend

Reference: `f09e46594`.

Adapt the same backend-resident loop contract used in Phase F3 rather than
creating a second incompatible implementation.

- Reuse generic speculative infrastructure where both architectures agree.
- Keep architecture-specific graph inputs and cache validation separate.
- Preserve adaptive draft limits, sampler fallbacks, and multi-slot isolation.

Commit: `glm-dsa: keep MTP draft state on backend`

Exit criterion: matching fallback output and acceptance, no supported-path
per-step host synchronization, and improved GLM-5.3 decode performance.

### D6. Reuse sparse masks across shared GLM-DSA indexers

Reference: `0b5c2f91a`.

Standard GLM-5.3 contains groups whose first layer has a full indexer and whose
following layers share its selection. Build the sparse mask once per compatible
group and reuse it without changing ownership or lifetime across graph rebuilds.

Commit: `glm-dsa: reuse sparse masks across shared indexers`

Exit criterion: matching logits versus per-layer mask construction, reduced
graph work, and passing tests for full/shared transitions and multiple streams.

### D7. Auto-fit and long-context validation

- Verify that target weights, the NextN layer, target context, and MTP context are
  counted exactly once by `common_fit_extra_model`.
- Test near the VRAM limit with MTP disabled and enabled.
- Confirm that `n_gpu_layers` placement includes all loaded NextN tensors.
- Change fit code only if a reproducible allocation gap remains; make that fix a
  separate commit with a focused regression test.

Potential commit, only if required: `fit: correct GLM-DSA NextN memory accounting`

Exit criterion: no post-fit OOM at the tested safety margin and no unnecessary
VRAM under-allocation.

## Deferred work

- Do not port `b028e9623` radix top-k unless profiling shows a regression on the
  actual deployment CUDA/toolchain and a new implementation benefits
  `topk = 2048`.
- Do not port the generic flash-attention top-k hint contract (`0be18c8ca`) into
  GLM5NEXT indexed attention. Re-evaluate it only if D1 still needs the contract
  for generic GLM-DSA FlashAttention.
- Do not apply shared-indexer mask reuse to GLM-5.3-Flash.
- Avoid duplicating the existing GLM5NEXT backend top-k cache, hidden buffers,
  auto-fit integration, and multi-slot state machinery.

## Final validation matrix

For both model families, run:

- MTP off and on.
- `n_slots = 1` and `n_slots = 2` or more.
- Short context, 50K, 80K, and 100K context.
- F16 and Q8_0 KV cache where supported.
- Greedy deterministic comparison plus the normal production sampler.
- Prompt reuse, LCP rewind, cancellation, sequence reset, and context shift.
- CUDA graph enabled and disabled.
- Auto-fit near the VRAM limit.

For GLM-5.3-Flash, also run multimodal prefill, repeated-image graph reuse, and
generation after multimodal prefill.

Record:

- Prompt tokens/s and per-chunk trend.
- Generated tokens/s and the trend as context grows.
- Draft acceptance and mean accepted length.
- GPU memory usage and allocation margin.
- CUDA graph reuse count.
- Any host synchronization or device-to-host transfer remaining in the hot path.

## Status checklist

- [ ] Phase 0 baseline recorded.
- [x] F1 GLM5NEXT regression coverage committed; awaiting review.
- [x] F2 redundant GLM5NEXT MTP indexer work removed and committed; awaiting review.
- [x] F3 backend-resident GLM5NEXT MTP loop committed and reviewed.
- [x] F4 GLM5NEXT validation checkpoint completed; awaiting review.
- [x] F5 multi-slot long-prompt scheduling fixed and committed; awaiting review.
- [ ] D0 GLM-DSA hunk-level audit completed.
- [ ] D1 GLM-DSA correctness foundation committed and reviewed.
- [ ] D2 safe full-indexer MTP reuse committed and reviewed.
- [ ] D3 backend GLM-DSA MTP top-k cache committed and reviewed.
- [ ] D4 reused GLM-DSA MTP indexer keys skipped, committed, and reviewed.
- [ ] D5 backend-resident GLM-DSA MTP loop committed and reviewed.
- [ ] D6 shared-indexer sparse-mask reuse committed and reviewed.
- [ ] D7 auto-fit and long-context validation completed.
- [ ] Final validation matrix completed and report updated.
